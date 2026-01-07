# import logging
# import os
# import cv2
# import numpy as np
# import time
# import torch
# import torch.nn as nn
# from utils.meter import AverageMeter
# from utils.metrics import R1_mAP_eval
# from torch.cuda import amp
# import torch.distributed as dist

# def do_train(cfg,
#              model,
#              center_criterion,
#              train_loader,
#              val_loader,
#              optimizer,
#              optimizer_center,
#              scheduler,
#              loss_fn,
#              num_query, local_rank):
#     log_period = cfg.SOLVER.LOG_PERIOD
#     checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
#     eval_period = cfg.SOLVER.EVAL_PERIOD

#     device = "cuda"
#     epochs = cfg.SOLVER.MAX_EPOCHS

#     logger = logging.getLogger("transreid.train")
#     logger.info('start training')
#     _LOCAL_PROCESS_GROUP = None
#     if device:
#         model.to(local_rank)
#         if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
#             logger.info('Using {} GPUs for training'.format(torch.cuda.device_count()))
#             model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

#     loss_meter = AverageMeter()
#     acc_meter = AverageMeter()

#     evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
#     scaler = amp.GradScaler()
#     # train
#     for epoch in range(1, epochs + 1):
#         start_time = time.time()
#         loss_meter.reset()
#         acc_meter.reset()
#         evaluator.reset()
#         model.train()
#         for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
#             optimizer.zero_grad()
#             optimizer_center.zero_grad()
#             img = img.to(device)
#             target = vid.to(device)
#             target_cam = target_cam.to(device)
#             target_view = target_view.to(device)
#             with amp.autocast(enabled=True):
#                 score, feat, _ = model(img, label=target, cam_label=target_cam, view_label=target_view )
#                 loss = loss_fn(score, feat, target, target_cam)

#             scaler.scale(loss).backward()

#             scaler.step(optimizer)
#             scaler.update()

#             if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
#                 for param in center_criterion.parameters():
#                     param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
#                 scaler.step(optimizer_center)
#                 scaler.update()
#             if isinstance(score, list):
#                 acc = (score[0].max(1)[1] == target).float().mean()
#             else:
#                 acc = (score.max(1)[1] == target).float().mean()

#             loss_meter.update(loss.item(), img.shape[0])
#             acc_meter.update(acc, 1)

#             torch.cuda.synchronize()
#             if cfg.MODEL.DIST_TRAIN:
#                 if dist.get_rank() == 0:
#                     if (n_iter + 1) % log_period == 0:
#                         base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
#                         logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
#                                     .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, base_lr))
#             else:
#                 if (n_iter + 1) % log_period == 0:
#                     base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
#                     logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
#                                 .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, base_lr))

#         end_time = time.time()
#         time_per_batch = (end_time - start_time) / (n_iter + 1)
#         if cfg.SOLVER.WARMUP_METHOD == 'cosine':
#             scheduler.step(epoch)
#         else:
#             scheduler.step()
#         if cfg.MODEL.DIST_TRAIN:
#             pass
#         else:
#             logger.info("Epoch {} done. Time per epoch: {:.3f}[s] Speed: {:.1f}[samples/s]"
#                     .format(epoch, time_per_batch * (n_iter + 1), train_loader.batch_size / time_per_batch))

#         if epoch % checkpoint_period == 0:
#             if cfg.MODEL.DIST_TRAIN:
#                 if dist.get_rank() == 0:
#                     torch.save(model.state_dict(),
#                                os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
#             else:
#                 torch.save(model.state_dict(),
#                            os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

#         if epoch % eval_period == 0:
#             if cfg.MODEL.DIST_TRAIN:
#                 if dist.get_rank() == 0:
#                     model.eval()
#                     for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
#                         with torch.no_grad():
#                             img = img.to(device)
#                             camids = camids.to(device)
#                             target_view = target_view.to(device)
#                             feat, _ = model(img, cam_label=camids, view_label=target_view)
#                             evaluator.update((feat, vid, camid))
#                     cmc, mAP, _, _, _, _, _ = evaluator.compute()
#                     logger.info("Validation Results - Epoch: {}".format(epoch))
#                     logger.info("mAP: {:.1%}".format(mAP))
#                     for r in [1, 5, 10]:
#                         logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
#                     torch.cuda.empty_cache()
#             else:
#                 model.eval()
#                 for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
#                     with torch.no_grad():
#                         img = img.to(device)
#                         camids = camids.to(device)
#                         target_view = target_view.to(device)
#                         feat, _ = model(img, cam_label=camids, view_label=target_view)
#                         evaluator.update((feat, vid, camid))
#                 cmc, mAP, _, _, _, _, _ = evaluator.compute()
#                 logger.info("Validation Results - Epoch: {}".format(epoch))
#                 logger.info("mAP: {:.1%}".format(mAP))
#                 for r in [1, 5, 10]:
#                     logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
#                 torch.cuda.empty_cache()





# def do_train_cl(cfg, 
#                 algorithm, 
#                 train_loader, 
#                 val_loader, 
#                 scheduler, 
#                 loss_fn, 
#                 num_query, 
#                 task_id):
    
#     log_period = cfg.SOLVER.LOG_PERIOD
#     eval_period = cfg.SOLVER.EVAL_PERIOD
#     epochs = cfg.SOLVER.MAX_EPOCHS

#     logger = logging.getLogger("transreid.train")
#     logger.info(f"Start CL Training Task {task_id+1} | Method: {algorithm.__class__.__name__}")

#     loss_meter = AverageMeter()
#     acc_meter = AverageMeter()

#     # 初始化评估器
#     evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

#     algorithm.model.train()
    
#     use_fp16 = getattr(cfg.SOLVER, 'FP16', False)
#     scaler = torch.cuda.amp.GradScaler() 

#     # === Epoch 循环 ===
#     for epoch in range(1, epochs + 1):
#         start_time = time.time()
#         loss_meter.reset()
#         acc_meter.reset()
#         evaluator.reset()
        
#         # === Batch 循环 ===
#         for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            
#             img = img.cuda(non_blocking=True)
#             target = vid.cuda(non_blocking=True)
#             target_cam = target_cam.cuda(non_blocking=True)
#             target_view = target_view.cuda(non_blocking=True)

#             # [核心修复] 闭包必须接收 **kwargs 以兼容 cam_label/view_label
#             def forwarding_closure(inputs, targets, **kwargs):
#                 # 1. 前向传播：将 kwargs (含 cam_label, view_label) 传给模型
#                 with torch.cuda.amp.autocast(enabled=use_fp16):
#                     outputs = algorithm.model(inputs, **kwargs) 
                    
#                     if isinstance(outputs, (tuple, list)):
#                         score, feat = outputs[0], outputs[1]
#                     else:
#                         score, feat = outputs, None
                    
#                     current_target_cam = kwargs.get('cam_label')
#                     total_loss = loss_fn(score, feat, targets, current_target_cam)
                
#                 return total_loss, score

#             # 调用算法的 observe
#             # 这里的 cam_label 会被打包进 kwargs 传给上面的 forwarding_closure
#             loss_value, cls_score = algorithm.observe(
#                 inputs=img, 
#                 targets=target, 
#                 task_loss_fn=forwarding_closure,
#                 scaler=scaler,
#                 cam_label=target_cam, 
#                 view_label=target_view
#             )
            
#             # 计算准确率
#             if isinstance(cls_score, list): cls_score = cls_score[0]
#             acc = (cls_score.max(1)[1] == target).float().mean()

#             loss_meter.update(loss_value, img.shape[0])
#             acc_meter.update(acc, 1)

#             if (n_iter + 1) % log_period == 0:
#                 current_lr = algorithm.optimizer.param_groups[0]['lr']
#                 logger.info(f"Epoch[{epoch}] Iter[{n_iter+1}/{len(train_loader)}] "
#                             f"Loss: {loss_meter.avg:.3f}, "
#                             f"Acc: {acc_meter.avg:.3f}, "
#                             f"Base Lr: {current_lr:.2e}")

#         # === Epoch 结束统计 ===
#         end_time = time.time()
#         time_per_epoch = end_time - start_time
#         speed = (len(train_loader) * train_loader.batch_size) / time_per_epoch
        
#         logger.info("Epoch {} done. Time per epoch: {:.3f}[s] Speed: {:.1f}[samples/s]"
#                     .format(epoch, time_per_epoch, speed))
        
#         scheduler.step(epoch)

#         # === 定期评估 ===
#         if epoch % eval_period == 0:
#             logger.info(f"Validation Results - Epoch: {epoch}")
#             algorithm.model.eval()
            
#             for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
#                 with torch.no_grad():
#                     img = img.cuda(non_blocking=True)
#                     camids = camids.cuda(non_blocking=True)
#                     target_view = target_view.cuda(non_blocking=True)
                    
#                     feat, _ = algorithm.model(img, cam_label=camids, view_label=target_view)
#                     evaluator.update((feat, vid, camid))
            
#             cmc, mAP, _, _, _, _, _ = evaluator.compute()
#             logger.info("mAP: {:.1%}".format(mAP))
#             for r in [1, 5, 10]:
#                 logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            
#             torch.cuda.empty_cache()
#             algorithm.model.train()

#     logger.info(f"Task {task_id+1} Finished. Running post-task routines...")
    
#     # 辅助函数：适配 task_end 钩子
#     def val_loss_helper(outputs, targets, target_cam):
#         if isinstance(outputs, (tuple, list)):
#             score, feat = outputs[0], outputs[1]
#         else:
#             score, feat = outputs, None
#         return loss_fn(score, feat, targets, target_cam)

#     algorithm.on_task_end(train_loader, val_loss_helper)
# def do_inference(cfg,
#                  model,
#                  val_loader,
#                  num_query):
#     device = "cuda"
#     logger = logging.getLogger("transreid.test")
#     logger.info("Enter inferencing")

#     evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING)

#     evaluator.reset()

#     if device:
#         if torch.cuda.device_count() > 1:
#             print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
#             model = nn.DataParallel(model)
#         model.to(device)

#     model.eval()
#     img_path_list = []

#     for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
#         with torch.no_grad():
#             img = img.to(device)
#             camids = camids.to(device)
#             target_view = target_view.to(device)
#             feat , _ = model(img, cam_label=camids, view_label=target_view)
#             evaluator.update((feat, pid, camid))
#             img_path_list.extend(imgpath)

#     cmc, mAP, _, _, _, _, _ = evaluator.compute()
#     logger.info("Validation Results ")
#     logger.info("mAP: {:.1%}".format(mAP))
#     for r in [1, 5, 10]:
#         logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
#     return cmc[0], cmc[4]


import logging
import os
import time
import torch
import torch.nn as nn
from torch.cuda import amp
import torch.distributed as dist

# [修正 Import] 指向 utils_i2i 文件夹，避免与 t2i 冲突
from utils.utils_i2i.meter import AverageMeter
from utils.utils_i2i.metrics import R1_mAP_eval

def do_train_i2i(cfg,
                 algorithm,       # [CL改动] 接收 algorithm 对象
                 train_loader,
                 val_loader,
                 scheduler,
                 loss_fn,
                 num_query,
                 task_id):        # [CL改动] 接收 task_id

    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info(f'Start CL Training Task {task_id}')
    
    # [CL改动] 获取模型 (algorithm.model 可能是 DDP 包装过的)
    model = algorithm.model

    # [CL改动] DDP 初始化通常在 main 函数做完了，这里不需要重复 init_process_group
    # 但我们可以保留 device 设置
    if device:
        # 如果 model 还没有在 device 上（虽然 CL 框架通常会处理）
        # model.to(device) 
        pass

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    
    # [CL改动] Scaler 通常由 algorithm 内部管理，或者我们需要传进去
    # 这里我们保留一个局部 scaler 传给 observe (如果 algorithm 支持)
    # 或者如果 algorithm.observe 内部处理了 scaler，这里就不需要了。
    # 为了兼容性，我们假设 algorithm.observe 接收 scaler 参数
    scaler = amp.GradScaler()

    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        
        # [CL改动] 确保模型处于训练模式
        model.train()
        
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            
            # [CL改动] 梯度清理由 algorithm 接管
            algorithm.optimizer.zero_grad()
            
            # [CL说明] Center Loss 的 optimizer 处理比较特殊
            # 如果 algorithm 不支持多 optimizer，我们暂时只能注释掉 optimizer_center 的逻辑
            # 或者将其放在 observe 之外手动 step（如果 center_loss 参数不在 model.parameters() 里）
            # optimizer_center.zero_grad() 
            
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)

            # === [CL核心改动] 定义闭包 (Forwarding Closure) ===
            # 持续学习算法需要重复调用这个函数来计算 Loss
            def forwarding_closure(inputs, targets, **kwargs):
                # 移除不必要的参数（防止 algorithm 传入多余 kwargs 报错）
                if 'optimizer' in kwargs: kwargs.pop('optimizer')
                if 'scaler' in kwargs: kwargs.pop('scaler')
                
                # 混合精度上下文
                with amp.autocast(enabled=True):
                    # 前向传播
                    outputs = model(inputs, cam_label=target_cam, view_label=target_view)
                    
                    if isinstance(outputs, (tuple, list)):
                        score, feat = outputs[0], outputs[1]
                    else:
                        score, feat = outputs, None
                    
                    # 计算 Loss
                    loss = loss_fn(score, feat, targets, target_cam)
                
                return loss, score

            # === [CL核心改动] 调用 Algorithm.observe ===
            # 替代了原来的 loss.backward() 和 optimizer.step()
            step_loss = algorithm.observe(
                inputs=img,
                targets=target,
                task_loss_fn=forwarding_closure,
                optimizer=algorithm.optimizer,
                scaler=scaler, # 传入 scaler 供内部 scaler.scale(loss).backward() 使用
                # 传入额外参数供 Closure 使用
                cam_label=target_cam,
                view_label=target_view
            )

            # === [CL适配] 解析返回值 ===
            # observe 通常返回 (loss, outputs) 或者只是 loss
            if isinstance(step_loss, (tuple, list)):
                loss_value = step_loss[0].item()
                score = step_loss[1]
            else:
                loss_value = step_loss.item() if isinstance(step_loss, torch.Tensor) else step_loss
                score = None # 这种情况下 Acc 可能无法精确统计，除非在 closure 里统计

            # [CL说明] 关于 Center Loss
            # 原代码逻辑：
            # if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
            #     for param in center_criterion.parameters(): ...
            #     scaler.step(optimizer_center)
            #     scaler.update()
            # 
            # 这里的难点是 center_criterion 没有传进来。
            # 如果必须保留 center loss，需要修改 do_train_cl 的接口把 optimizer_center 传进来。
            # 鉴于 CL 场景通常会简化 Loss，这里暂时略过 Center Loss 的显式 Step。
            # 如果 algorithm.optimizer 包含了 center param，则已经被更新了。

            # 计算 Acc
            acc = 0
            if score is not None:
                if isinstance(score, list):
                    acc = (score[0].max(1)[1] == target).float().mean()
                else:
                    acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss_value, img.shape[0])
            acc_meter.update(acc, 1)

            # 日志打印 (保持原样)
            if (n_iter + 1) % log_period == 0:
                # 获取当前 LR
                try:
                    base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                except:
                    base_lr = algorithm.optimizer.param_groups[0]['lr']

                logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, base_lr))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        
        # Scheduler Step
        if cfg.SOLVER.WARMUP_METHOD == 'cosine':
            scheduler.step(epoch)
        else:
            scheduler.step()

        logger.info("Epoch {} done. Time per epoch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch * (n_iter + 1), train_loader.batch_size / time_per_batch))

        # Checkpoint (适配路径)
        if epoch % checkpoint_period == 0:
            # 简单处理分布式 rank
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        # Validation (保持原样，直接复用)
        if epoch % eval_period == 0:
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat, _ = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()
            
            # 如果是 DDP，最好 barrier 一下
            if dist.is_initialized():
                dist.barrier()

# 推理函数保持不变，只需修正 import
def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat , _ = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]