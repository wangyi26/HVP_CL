import logging
import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.cuda import amp
import torch.distributed as dist

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
    
    image_feats = torch.nn.functional.normalize(image_feats, dim=-1, p=2)
    text_feats = torch.nn.functional.normalize(text_feats, dim=-1, p=2)
    
    sims = torch.matmul(text_feats, image_feats.t())
    
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
        
    r1 = correct_1 / len(text_pids)
    r5 = correct_5 / len(text_pids)
    r10 = correct_10 / len(text_pids)
    
    logger.info(f"T2I Result: R1: {r1:.1%}, R5: {r5:.1%}, R10: {r10:.1%}")
    return r1, r5

def do_train_cl(cfg,
                algorithm,      # [注意] 这里接收的是 algorithm 对象
                train_loader,
                val_loader,
                scheduler,
                loss_fn,        # [注意] 这里必须接收 loss_fn
                num_query,
                task_id):
    
    log_period = cfg.SOLVER.LOG_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("cl_core.train")
    logger.info(f'Start Continual Learning Training Task {task_id}')

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    # 获取 model 引用 (algorithm 内部持有 model)
    model = algorithm.model
    # 自动识别当前任务类型
    current_task_type = model.task_types.get(model.current_task, 'reid')

    use_fp16 = getattr(cfg.SOLVER, 'FP16', False)
    scaler = torch.cuda.amp.GradScaler() 

    # Train
    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        acc_meter.reset()
        model.train()
        
        for n_iter, batch in enumerate(train_loader):
            
            # ==================================================
            # 1. 统一数据解包
            # ==================================================
            if isinstance(batch, dict):
                # T2I (IRRA)
                inputs = batch['images'].to(device)
                targets = batch['pids'].to(device)
                text_inputs = batch['caption_ids'].to(device)
                camid = None
                viewid = None
            else:
                # I2I (TransReID)
                # loader 返回 (imgs, pids, camids, viewids)
                inputs = batch[0].to(device)
                targets = batch[1].to(device)
                camid = batch[2].to(device) if len(batch) > 2 else None
                viewid = batch[3].to(device) if len(batch) > 3 else None
                text_inputs = None

            algorithm.optimizer.zero_grad()
            
            # ==================================================
            # 2. 定义闭包 (Forwarding Closure)
            #    关键：必须返回 (loss, score) 两个值
            # ==================================================
            def forwarding_closure(inputs, targets, text=None, **kwargs):
                # [Fix 1] 移除 algorithm 传进来的不需要的参数
                if 'optimizer' in kwargs: kwargs.pop('optimizer')
                if 'scaler' in kwargs: kwargs.pop('scaler')
                
                # --- T2I 分支 ---
                if current_task_type == 't2i':
                    outputs = model(inputs, text=text, label=targets, **kwargs)
                    if isinstance(outputs, dict):
                        loss = outputs.get('loss', 0.0)
                        acc = outputs.get('img_acc', 0.0)
                    else:
                        loss = outputs
                        acc = 0.0
                    return loss, acc
                
                # --- I2I 分支 ---
                else:
                    # 补充 cam_label / view_label
                    forward_kwargs = kwargs.copy()
                    if 'cam_label' not in forward_kwargs and camid is not None:
                        forward_kwargs['cam_label'] = camid
                    if 'view_label' not in forward_kwargs and viewid is not None:
                        forward_kwargs['view_label'] = viewid

                    with torch.cuda.amp.autocast(enabled=use_fp16):
                        # 1. 模型前向
                        outputs = model(inputs, **forward_kwargs)
                        
                        # 2. 解析输出 (score, feat, ...)
                        if isinstance(outputs, (tuple, list)):
                            score, feat = outputs[0], outputs[1]
                        else:
                            score, feat = outputs, None
                        
                        # 3. [Fix 2] 在闭包内计算 Loss
                        current_target_cam = forward_kwargs.get('cam_label')
                        
                        # 调用传入的 loss_fn
                        total_loss = loss_fn(score, feat, targets, current_target_cam)
                
                # 4. [Fix 3] 严格返回两个值: (loss, score)
                # 这样 finetuning.py 里的 `loss, score = task_loss_fn(...)` 才能解包成功
                return total_loss, score

            # ==================================================
            # 3. 执行算法步
            # ==================================================
            step_loss = algorithm.observe(
                inputs, targets, forwarding_closure, 
                text=text_inputs,
                optimizer=algorithm.optimizer,
                # 传递元数据供 EWC 等使用
                cam_label=camid,
                view_label=viewid
            )

            # 更新日志
            if isinstance(step_loss, (tuple, list)):
                main_loss = step_loss[0]
            else:
                main_loss = step_loss
            
            # 判断是否需要调用 .item()
            if isinstance(main_loss, torch.Tensor):
                loss_val = main_loss.item()
            else:
                loss_val = main_loss
                
            loss_meter.update(loss_val, inputs.shape[0])
            
            # 计算 Accuracy (仅 I2I)
            if current_task_type != 't2i':
                # 重新计算一次 forward 用于 acc 显示? 
                # 或者 algorithm.observe 返回了 score? 
                # 标准 Finetuning observe 返回 (loss, outputs)
                # 如果 step_loss 是 tuple (loss, outputs)
                if isinstance(step_loss, (tuple, list)) and len(step_loss) >= 2:
                    outputs = step_loss[1]
                    if isinstance(outputs, torch.Tensor):
                        acc = (outputs.max(1)[1] == targets).float().mean()
                        acc_meter.update(acc, 1)

            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, 
                                    algorithm.optimizer.param_groups[0]['lr']))

        # === Epoch End ===
        scheduler.step(epoch) 

        # === Evaluation ===
        if epoch % eval_period == 0 and val_loader is not None:
            logger.info("Validation Epoch {}".format(epoch))
            if current_task_type == 't2i':
                eval_t2i(model, val_loader, device)
            else:
                do_inference(cfg, model, val_loader, num_query)
            
            # 确保切回训练模式
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
            
            # eval 模式下 wrapper 返回 (feat, featmaps)
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