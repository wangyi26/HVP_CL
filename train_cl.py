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
    if not os.path.exists(output_dir):os.makedirs(output_dir)
    
    cleanup_logger("cl_core")
    logger = setup_logger("cl_core", output_dir, 1)
    
    logger.info(f"\n{'='*20} Start Training Task {task_idx}: {task_name} ({task_type}) {'='*20}")
    logger.info(f"Using Config File: {config_file}")
    if args.opts:
        logger.info(f"Overriding config with opts: {args.opts}")
    logger.info(f"Loaded configuration: \n{cfg}")
    
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for training")

    # -------------------------------------------------------------------------
    # 2. 准备数据 (获取 Dataset 对象以读取 num_classes/cameras/views)
    # -------------------------------------------------------------------------
    logger.info("Preparing dataset...")
    dataset = get_dataset(task_name, cfg.DATASETS.ROOT_DIR)
    
    # 获取必要的元数据
    num_classes = dataset.num_train_pids
    # 注意：I2I 需要 camera/view 信息用于 SIE，T2I 不需要但可以保留占位
    cam_num = dataset.num_train_cams if hasattr(dataset, 'num_train_cams') else 0
    view_num = dataset.num_train_vids if hasattr(dataset, 'num_train_vids') else 0

    logger.info(f"Dataset {task_name}: {num_classes} classes, {cam_num} cameras, {view_num} views")

    # 构建 DataLoader
    train_loader = build_dataloader(cfg, dataset, task_type=task_type, is_train=True)
    
    # 验证集 (可选)
    val_loader = None
    if cfg.SOLVER.EVAL_PERIOD > 0:
         val_loader = build_dataloader(cfg, dataset, task_type=task_type, is_train=False)

    # -------------------------------------------------------------------------
    # 3. 构建/继承模型
    # -------------------------------------------------------------------------
    if prev_model is None:
        logger.info("Building NEW model...")
        
        # === [核心修复] 分支构建基础模型 ===
        if task_type == 'i2i':
            # TransReID make_model 需要 camera_num 和 view_num
            base_model = make_i2i_model(
                cfg, 
                num_class=num_classes, 
                camera_num=cam_num, 
                view_num=view_num,
                semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT
            )
            # ViT-Base 默认 768, Small 384. 从 cfg 读取更安全
        else:
            # IRRA build_model 只需要 num_classes
            base_model = make_t2i_model(cfg, num_classes=num_classes)

        if hasattr(base_model, 'in_planes'):
            # Case 1: TransReID / ResNet 标准属性
            feat_dim = base_model.in_planes
        elif hasattr(base_model, 'base') and hasattr(base_model.base, 'num_features'):
            # Case 2: 你的旧逻辑 (针对 ViT backbone)
            feat_dim = base_model.base.num_features[-1]
        elif hasattr(base_model, 'embed_dim'):
            # Case 3: IRRA / CLIP 常用属性
            feat_dim = base_model.embed_dim
        else:
            # Case 4: 兜底 (从配置读取)
            logger.warning("Could not auto-detect feature dim, falling back to cfg.MODEL.FEATURE_DIM")
            feat_dim = cfg.MODEL.FEATURE_DIM
            
        # 使用 Universal Wrapper 包装
        model = UniversalCLModel(base_model, feature_dim=feat_dim, task_type=task_type)
        model.cuda()
        
    else:
        logger.info("Inheriting PREVIOUS model...")
        model = prev_model
    
    # 激活当前任务 Head
    model.add_task(task_name, num_classes=num_classes, task_type=task_type)
    model.set_current_task(task_name)
    model.cuda()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Size: {num_params / 1e6:.2f}M params")

    # -------------------------------------------------------------------------
    # 4. 准备 Loss, Optimizer, Scheduler
    # -------------------------------------------------------------------------
    loss_func = None
    center_criterion = None
    
    if task_type == 'i2i':
        # I2I: 使用外部 Loss 函数
        loss_func, center_criterion = make_i2i_loss(cfg, num_classes=num_classes)
    else:
        # T2I: Loss 集成在模型内部，不需要外部 Loss Function
        logger.info("Using internal T2I loss (SDM+MLM+ID)")

    # 优化器
    optimizer, optimizer_center = make_cl_optimizer(cfg, model, center_criterion)
    
    # 学习率调度器 (I2I/T2I 通用)
    scheduler = create_scheduler_i2i(cfg, optimizer)

    # -------------------------------------------------------------------------
    # 5. 构建算法 (EWC / Finetuning)
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
        
        # 定义 Loss Helper 用于 Fisher 计算
        def val_loss_helper(outputs, targets, **kwargs):
            if task_type == 't2i':
                # T2I 返回 dict
                return outputs['loss'] if isinstance(outputs, dict) else outputs
            else:
                # I2I 返回 (score, feat)
                # 注意：make_loss 里的 loss_func 通常需要 (score, feat, target)
                if isinstance(outputs, (tuple, list)):
                    score, feat = outputs[0], outputs[1]
                else:
                    score, feat = outputs, None
                return loss_func(score, feat, targets)
        
        logger.info("Calculating Fisher Matrix for pretrained model...")
        algorithm.on_task_end(train_loader, val_loss_helper)
        logger.info("Fisher Matrix calculated.")
        
    else:
        # 正常训练
        do_train_cl(
            cfg,
            algorithm,      # 1. 传入算法对象 (内部包含 model 和 optimizer)
            train_loader,   # 2. 训练数据
            val_loader,     # 3. 验证数据
            scheduler,      # 4. 调度器 (之前漏传了)
            loss_func,      # 5. Loss 函数 (之前漏传了)
            0,              # 6. num_query (占位，训练时仅打印用)
            task_idx        # 7. 任务 ID
        )
        # 训练结束后 algorithm 会自动调用 on_task_end
    
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