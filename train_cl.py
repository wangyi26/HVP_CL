import sys
import os
import argparse
import torch
import copy
import random
import numpy as np
import logging
import time


# === 2. 引入配置与基础工具 ===
from config import cfg
from utils.logger import setup_logger


# === 3. 引入核心模块 (CL Core) ===
from cl_core.model_wrapper import UniversalCLModel
from cl_core.algorithms import build_algorithm
from cl_core.processor import do_train_cl
# 引入之前移出去的统一优化器构建函数
from cl_core.solver.build import make_cl_optimizer 

# === 4. 引入数据与模型工厂 ===
from datasets.factory import get_dataset
from datasets.loader import build_dataloader

# === 5. 引入外部模型构建器 ===
# I2I (TransReID)
from external_models.transreid.make_model import make_model as make_i2i_model
from external_models.transreid.loss.make_loss import make_loss as make_i2i_loss
from external_models.transreid.solver import create_scheduler as create_scheduler_i2i
# T2I (IRRA)
from external_models.irra.build import build_model as make_t2i_model

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def reset_config(config_file, opts=None):
    cfg.defrost()
    cfg.merge_from_file(config_file)
    if opts:
        cfg.merge_from_list(opts)
    cfg.freeze()

def cleanup_logger(logger_name):
    logger = logging.getLogger(logger_name)
    while logger.handlers:
        logger.removeHandler(logger.handlers[0])

def train_one_task(task_idx, task_info, args, prev_model=None, prev_alg=None):
    task_name = task_info['name']
    task_type = task_info['type']
    config_file = task_info['config']
    
    # -------------------------------------------------------------------------
    # 1. 配置与日志
    # -------------------------------------------------------------------------
    reset_config(config_file, args.opts)
    
    output_dir = os.path.join(cfg.OUTPUT_DIR, f"task{task_idx}_{task_name}")
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    cleanup_logger("cl_core")
    logger = setup_logger("cl_core", output_dir, 1)
    
    logger.info(f"\n{'='*20} Start Training Task {task_idx}: {task_name} ({task_type}) {'='*20}")
    logger.info(f"Using Config File: {config_file}")
    if args.opts:
        logger.info(f"Overriding config with opts: {args.opts}")
    
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for training")

    # -------------------------------------------------------------------------
    # 2. 准备数据
    # -------------------------------------------------------------------------
    logger.info("Preparing dataset...")
    dataset = get_dataset(task_name, cfg.DATASETS.ROOT_DIR)
    
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams if hasattr(dataset, 'num_train_cams') else 0
    view_num = dataset.num_train_vids if hasattr(dataset, 'num_train_vids') else 0

    logger.info(f"Dataset {task_name}: {num_classes} classes, {cam_num} cameras, {view_num} views")

    train_loader = build_dataloader(cfg, dataset, task_type=task_type, is_train=True)
    val_loader = None
    if cfg.SOLVER.EVAL_PERIOD > 0:
         val_loader = build_dataloader(cfg, dataset, task_type=task_type, is_train=False)

    # -------------------------------------------------------------------------
    # 3. 构建/继承模型 (核心修改区域)
    # -------------------------------------------------------------------------
    if prev_model is None:
        logger.info("Initializing Shared Visual Backbone...")
        
        # [修改点 1] 临时构建一个任务模型，目的是“拆”出它的 Visual Backbone
        if task_type == 'i2i':
            # I2I: TransReID
            temp_model = make_i2i_model(
                cfg, 
                num_class=num_classes, 
                camera_num=cam_num, 
                view_num=view_num,
                semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT
            )
            # 提取 ViT Backbone (通常是 model.base)
            visual_backbone = temp_model.base
            feat_dim = temp_model.in_planes
            
        else: # t2i
            # T2I: IRRA
            temp_model = make_t2i_model(cfg, num_classes=num_classes)
            # 提取 ViT Backbone (IRRA 中通常是 model.vis_model)
            visual_backbone = temp_model.vis_model
            feat_dim = temp_model.embed_dim

        # [修改点 2] 实例化 UniversalCLModel，只传入 Visual Backbone
        # 注意：不再传入完整的 base_model，而是 backbone
        model = UniversalCLModel(visual_backbone, feature_dim=feat_dim, task_type=task_type)
        model.cuda()
        
        # 此时 UniversalCLModel 会自动将这个 Backbone 的 Input Layer 注册为 'default' Stem
        
    else:
        logger.info("Inheriting PREVIOUS model...")
        model = prev_model
    
    # [修改点 3] 激活当前任务，务必传入 cfg！
    # UniversalCLModel 需要 cfg 来判断分辨率是否变化 (Resize Stem) 以及构建 T2I Head
    model.add_task(task_name, num_classes=num_classes, task_type=task_type, cfg=cfg)
    model.set_current_task(task_name)
    model.cuda()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Size (Trainable): {num_params / 1e6:.2f}M params")

    # -------------------------------------------------------------------------
    # 4. 准备 Loss, Optimizer, Scheduler
    # -------------------------------------------------------------------------
    loss_func = None
    center_criterion = None
    
    if task_type == 'i2i':
        loss_func, center_criterion = make_i2i_loss(cfg, num_classes=num_classes)
    else:
        logger.info("Using internal T2I loss (SDM+MLM+ID)")

    # 优化器 (make_cl_optimizer 应该已经适配了 T2I 参数分组)
    optimizer, optimizer_center = make_cl_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler_i2i(cfg, optimizer)

    # -------------------------------------------------------------------------
    # 5. 构建算法
    # -------------------------------------------------------------------------
    algorithm = build_algorithm(
        cfg.CL.METHOD, 
        model, 
        optimizer, 
        ewc_lambda=cfg.CL.EWC_LAMBDA,
        prev_algorithm=prev_alg
    )

    # -------------------------------------------------------------------------
    # 6. 训练或加载预训练 Fisher
    # -------------------------------------------------------------------------
    if task_info.get('load_pretrained', False):
        weights_path = task_info['weights']
        logger.info(f"Loading pretrained weights from: {weights_path}")
        model.load_state_dict(torch.load(weights_path), strict=False)
        
        def val_loss_helper(outputs, targets, **kwargs):
            if task_type == 't2i':
                return outputs['loss'] if isinstance(outputs, dict) else outputs
            else:
                if isinstance(outputs, (tuple, list)):
                    score, feat = outputs[0], outputs[1]
                else:
                    score, feat = outputs, None
                return loss_func(score, feat, targets)
        
        logger.info("Calculating Fisher Matrix for pretrained model...")
        algorithm.on_task_end(train_loader, val_loss_helper)
        
    else:
        do_train_cl(
            cfg,
            algorithm,
            train_loader,
            val_loader,
            scheduler,
            loss_func,
            0,
            task_idx
        )
    
    # 保存结果
    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pth"))
    
    return model, algorithm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Continual Learning Training")
    parser.add_argument("--config_file", default="", help="Global config file", type=str)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    set_seed(1234)
    
    # 定义任务流
    TASKS = [
        # Task 1: Market1501 (I2I)
        # {
        #     'name': 'Market1501',
        #     'type': 'i2i',
        #     'config': 'configs/market1501/vit_tiny.yml',
        #     'load_pretrained': False
        # },
        # Task 2: MSMT17 I2I)
        # {
        #     'name': 'MSMT17',
        #     'type': 'i2i',
        #     'config': 'configs/msmt17/vit_tiny.yml',
        #     'load_pretrained': False
        # },
        # Task 3:cuhkpedes T2I
        # {
        #     'name': 'CUHK-PEDES',
        #     'type': 't2i',
        #     'config': 'configs/cuhkpedes/vit_tiny.yml',
        #     'load_pretrained': False
        # },
        # Task 4: icfgpedes T2I
        {   
            'name': 'ICFG-PEDES',
            'type': 't2i',
            'config': 'configs/icfgpedes/vit_tiny.yml',
            'load_pretrained': False
        }
    ]

    curr_model, curr_alg = None, None
    
    for i, t in enumerate(TASKS):
        curr_model, curr_alg = train_one_task(i+1, t, args, curr_model, curr_alg)