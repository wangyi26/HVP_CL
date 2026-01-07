import logging
import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.cuda import amp
import torch.distributed as dist
from collections import defaultdict
from prettytable import PrettyTable

from utils.meter import AverageMeter
from utils.metrics.metrics_i2i import R1_mAP_eval

@torch.no_grad()
def eval_t2i(model, val_loader, device="cuda"):
    """
    IRRA (Text-to-Image) 专用评估函数
    """
    model.eval()
    logger = logging.getLogger("cl_core.test")
    
    image_feats = []
    image_pids = []
    text_feats = []
    text_pids = []
    
    # === 1. 提取特征 ===
    for batch in val_loader:
        if isinstance(batch, dict):
            images = batch.get('images')
            texts = batch.get('caption_ids')
            pids = batch.get('pids')
            
            if images is not None:
                images = images.to(device)
                img_feat = model.base_model.encode_image(images)
                image_feats.append(img_feat.cpu())
                image_pids.extend(pids.tolist())
                
            if texts is not None:
                texts = texts.to(device)
                txt_feat = model.base_model.encode_text(texts)
                text_feats.append(txt_feat.cpu())
                text_pids.extend(pids.tolist())
    
    image_feats = torch.cat(image_feats, dim=0)
    text_feats = torch.cat(text_feats, dim=0)
    
    # === 2. 计算相似度 ===
    image_feats = torch.nn.functional.normalize(image_feats, dim=-1, p=2)
    text_feats = torch.nn.functional.normalize(text_feats, dim=-1, p=2)
    
    sims = torch.matmul(text_feats, image_feats.t())
    
    # === 3. 计算指标 (R1, R5, R10) ===
    topk_idx = sims.topk(k=10, dim=1)[1]
    topk_idx = topk_idx.numpy()
    text_pids = np.array(text_pids)
    image_pids = np.array(image_pids)
    
    correct_1 = 0
    correct_5 = 0
    correct_10 = 0
    
    for i, txt_pid in enumerate(text_pids):
        retrieved_pids = image_pids[topk_idx[i]]
        if txt_pid in retrieved_pids[:1]: correct_1 += 1
        if txt_pid in retrieved_pids[:5]: correct_5 += 1
        if txt_pid in retrieved_pids[:10]: correct_10 += 1
        
    r1 = 100.0 * correct_1 / len(text_pids)
    r5 = 100.0 * correct_5 / len(text_pids)
    r10 = 100.0 * correct_10 / len(text_pids)
    
    # mAP 和 mINP 简单占位 (完整计算比较耗时，这里主要复现 R1/R5/R10)
    # 如果需要完整 mAP，需要引入 metric_learning.py 里的计算逻辑
    map_score = 0.0 
    minp_score = 0.0

    # === 4. 打印表格 (复现原始日志风格) ===
    table = PrettyTable(['task', 'R1', 'R5', 'R10', 'mAP', 'mINP'])
    table.add_row(['t2i', f'{r1:.3f}', f'{r5:.3f}', f'{r10:.3f}', f'{map_score:.3f}', f'{minp_score:.3f}'])
    
    logger.info("Validation Results - T2I")
    logger.info("\n" + str(table))
    
    return r1, r5

def do_train_cl(cfg,
                algorithm,      
                train_loader,
                val_loader,
                scheduler,
                loss_fn,        
                num_query,
                task_id):
    
    log_period = cfg.SOLVER.LOG_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("cl_core.train")
    logger.info(f'Start Continual Learning Training Task {task_id}')

    # === I2I Meters ===
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    # === T2I Meters (动态管理所有 loss 和 acc) ===
    t2i_meters = defaultdict(AverageMeter)
    
    model = algorithm.model
    current_task_type = model.task_types.get(model.current_task, 'reid')

    use_fp16 = getattr(cfg.SOLVER, 'FP16', False)
    scaler = torch.cuda.amp.GradScaler() 

    # Train
    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        acc_meter.reset()
        # 重置所有 T2I meters
        for k in t2i_meters: t2i_meters[k].reset()
        
        model.train()
        
        for n_iter, batch in enumerate(train_loader):
            
            # 1. 统一数据解包
            t2i_kwargs = {} 
            if isinstance(batch, dict):
                # T2I (IRRA)
                inputs = batch['images'].to(device)
                targets = batch['pids'].to(device)
                text_inputs = batch['caption_ids'].to(device)
                # 动态提取所有剩余的键
                for k, v in batch.items():
                    if k not in ['images', 'pids', 'caption_ids']:
                        if isinstance(v, torch.Tensor):
                            t2i_kwargs[k] = v.to(device)
                        else:
                            t2i_kwargs[k] = v
                camid = None
                viewid = None
            else:
                # I2I (TransReID)
                inputs = batch[0].to(device)
                targets = batch[1].to(device)
                camid = batch[2].to(device) if len(batch) > 2 else None
                viewid = batch[3].to(device) if len(batch) > 3 else None
                text_inputs = None

            algorithm.optimizer.zero_grad()
            
            # 用于在闭包内捕获当前 batch 的详细指标 (供日志使用)
            batch_metrics_cache = {}

            # 2. 定义闭包
            def forwarding_closure(inputs, targets, text=None, **kwargs):
                if 'optimizer' in kwargs: kwargs.pop('optimizer')
                if 'scaler' in kwargs: kwargs.pop('scaler')
                
                # --- T2I 分支 ---
                if current_task_type == 't2i':
                    outputs = model(inputs, text=text, label=targets, **kwargs)
                    
                    if isinstance(outputs, dict):
                        # 1. 自动汇总 Loss (用于反向传播)
                        losses = [v for k, v in outputs.items() if 'loss' in k and isinstance(v, torch.Tensor)]
                        if len(losses) > 0:
                            loss = sum(losses)
                        else:
                            loss = torch.tensor(0.0, device=device, requires_grad=True)
                        
                        # 2. [关键] 捕获所有详细指标用于日志 (loss, acc 等)
                        # 将 Tensor 转为 Python float 存入 cache
                        for k, v in outputs.items():
                            if isinstance(v, torch.Tensor) and v.numel() == 1:
                                batch_metrics_cache[k] = v.item()
                            elif isinstance(v, (int, float)):
                                batch_metrics_cache[k] = v
                        
                        # 也记录总 Loss
                        batch_metrics_cache['loss'] = loss.item()
                            
                        acc = outputs.get('img_acc', 0.0)
                    else:
                        loss = outputs
                        acc = 0.0
                    return loss, acc
                
                # --- I2I 分支 ---
                else:
                    forward_kwargs = kwargs.copy()
                    if 'cam_label' not in forward_kwargs and camid is not None:
                        forward_kwargs['cam_label'] = camid
                    if 'view_label' not in forward_kwargs and viewid is not None:
                        forward_kwargs['view_label'] = viewid

                    with torch.cuda.amp.autocast(enabled=use_fp16):
                        outputs = model(inputs, **forward_kwargs)
                        if isinstance(outputs, (tuple, list)):
                            score, feat = outputs[0], outputs[1]
                        else:
                            score, feat = outputs, None
                        
                        current_target_cam = forward_kwargs.get('cam_label')
                        total_loss = loss_fn(score, feat, targets, current_target_cam)
                
                return total_loss, score

            # 3. 执行算法步
            step_loss = algorithm.observe(
                inputs, targets, forwarding_closure, 
                text=text_inputs,
                optimizer=algorithm.optimizer,
                cam_label=camid,
                view_label=viewid,
                **t2i_kwargs 
            )

            # 4. 更新 Meters
            if current_task_type == 't2i':
                # 更新 T2I 的详细 meters
                for k, v in batch_metrics_cache.items():
                    t2i_meters[k].update(v, inputs.shape[0])
            else:
                # 更新 I2I 的标准 meters
                if isinstance(step_loss, (tuple, list)):
                    main_loss = step_loss[0]
                else:
                    main_loss = step_loss
                
                loss_val = main_loss.item() if isinstance(main_loss, torch.Tensor) else main_loss
                loss_meter.update(loss_val, inputs.shape[0])
                
                if isinstance(step_loss, (tuple, list)) and len(step_loss) >= 2:
                    outputs = step_loss[1]
                    if isinstance(outputs, torch.Tensor):
                        acc = (outputs.max(1)[1] == targets).float().mean()
                        acc_meter.update(acc, 1)

            # 5. [核心修改] 分类打印日志
            if (n_iter + 1) % log_period == 0:
                lr = algorithm.optimizer.param_groups[0]['lr']
                
                if current_task_type == 't2i':
                    # === IRRA 风格日志 ===
                    # 格式: Epoch[34] Iteration[100/271], loss: 4.9062, sdm_loss: 3.4116, ...
                    log_msg = "Epoch[{}] Iteration[{}/{}]".format(epoch, (n_iter + 1), len(train_loader))
                    
                    # 优先打印关键指标
                    priority_keys = ['loss', 'sdm_loss', 'id_loss', 'mlm_loss', 'img_acc', 'txt_acc', 'mlm_acc']
                    for k in priority_keys:
                        if k in t2i_meters:
                            log_msg += ", {}: {:.4f}".format(k, t2i_meters[k].avg)
                    
                    # 打印其他可能存在的指标
                    for k in t2i_meters:
                        if k not in priority_keys:
                            log_msg += ", {}: {:.4f}".format(k, t2i_meters[k].avg)
                            
                    log_msg += ", Base Lr: {:.2e}".format(lr)
                    logger.info(log_msg)
                    
                else:
                    # === I2I 风格日志 ===
                    logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader),
                                        loss_meter.avg, acc_meter.avg, lr))

        # === Epoch End ===
        scheduler.step(epoch) 

        # === Evaluation ===
        if epoch % eval_period == 0 and val_loader is not None:
            logger.info(f"Validation Epoch {epoch}")
            if current_task_type == 't2i':
                eval_t2i(model, val_loader, device)
            else:
                do_inference(cfg, model, val_loader, num_query)
            
            model.train()

def do_inference(cfg, model, val_loader, num_query):
    device = "cuda"
    logger = logging.getLogger("cl_core.test")
    logger.info("Enter inferencing (I2I)")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING)
    evaluator.reset()

    model.eval()
    
    for n_iter, batch in enumerate(val_loader):
        with torch.no_grad():
            img = batch[0].to(device)
            pid = batch[1]
            camid = batch[2]
            camids = batch[3].to(device)
            target_view = batch[4].to(device)
            
            ret = model(img, cam_label=camids, view_label=target_view)
            
            if isinstance(ret, tuple):
                feat = ret[0]
            else:
                feat = ret
                
            evaluator.update((feat, pid, camid))
            
    cmc, mAP = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]