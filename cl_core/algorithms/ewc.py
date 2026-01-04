# import torch
# from .base import BaseCLAlgorithm

# class EWC(BaseCLAlgorithm):
#     def __init__(self, model, optimizer, ewc_lambda=1000, **kwargs):
#         super().__init__(model, optimizer, **kwargs)
#         self.ewc_lambda = ewc_lambda
        
#         # 存储旧任务的 Fisher Matrix 和 参数
#         # 结构: {param_name: tensor}
#         self.fisher = {} 
#         self.params = {} 
        
#         # 只对 Backbone 做 EWC 保护，忽略 Heads
#         self.valid_keys = [] 

#     def observe(self, inputs, targets, task_loss_fn, **kwargs):
#         self.optimizer.zero_grad()
        
#         #outputs = self.model(inputs, **kwargs)
#         loss, score = task_loss_fn(inputs, targets, **kwargs)
        
#         # === 添加 EWC 正则项 ===
#         if len(self.fisher) > 0:
#             ewc_loss = 0
#             for n, p in self.model.backbone.named_parameters():
#                 if n in self.valid_keys:
#                     # Loss += (lambda/2) * Fisher * (theta - theta_old)^2
#                     # 这里的 fisher 已经包含了 1/N 归一化，系数可根据论文微调
#                     _loss = self.fisher[n] * (p - self.params[n]).pow(2)
#                     ewc_loss += _loss.sum()
            
#             loss += self.ewc_lambda * ewc_loss
            
#         loss.backward()
#         self.optimizer.step()
        
#         return loss.item(), score

#     def on_task_end(self, dataloader, task_loss_fn, **kwargs):
#         """计算当前任务的 Fisher Matrix 并累积"""
#         print("==> [EWC] Calculating Fisher Matrix for current task...")
        
#         # 切换到 eval 模式，但我们要计算梯度，所以不用 torch.no_grad()
#         # 注意：有些层(如BN)在 eval 模式下的行为不同，通常计算 Fisher 时保持 eval 是合理的
#         self.model.train()
        
#         fisher_current = {}
        
#         # 初始化 Fisher 累加器
#         for n, p in self.model.backbone.named_parameters():
#             if p.requires_grad:
#                 fisher_current[n] = torch.zeros_like(p.data)
        
#         # 遍历数据计算梯度
#         num_samples = 0
        
#         for batch in dataloader:
#             # [核心修复] 正确解包 4 个返回值
#             # train_loader 返回: (img, pid, camid, viewid)
#             inputs, targets, target_cam, target_view = batch
            
#             inputs = inputs.to(self.device)
#             targets = targets.to(self.device)
#             target_cam = target_cam.to(self.device)
#             target_view = target_view.to(self.device)
            
#             self.model.zero_grad()
            
#             # [核心修复] 传递 cam_label 和 view_label 给模型 (为了 SIE)
#             # UniversalCLModel 会把 kwargs 传给 backbone
#             outputs = self.model(inputs, cam_label=target_cam, view_label=target_view)
            
#             # [核心修复] 传递 target_cam 给 loss helper
#             loss = task_loss_fn(outputs, targets, target_cam)
#             loss.backward()
            
#             # 累积梯度的平方
#             for n, p in self.model.backbone.named_parameters():
#                 if p.grad is not None:
#                     fisher_current[n] += p.grad.data.pow(2)
            
#             num_samples += 1
#             # 可选：如果数据集太大，可以限制样本数以加快计算
#             # if num_samples > 200: break

#         # 归一化
#         for n in fisher_current:
#             fisher_current[n] /= num_samples
        
#         # 更新 Fisher 和 参数副本
#         # 简单的 EWC 策略：直接存储当前任务结束时的参数作为 anchor
#         for n, p in self.model.backbone.named_parameters():
#             if p.requires_grad:
#                 self.params[n] = p.data.clone()
        
#         # 更新 Fisher Matrix
#         # (进阶策略可以是加权平均，这里简化为直接覆盖或相加，视具体 EWC 变种而定)
#         # 这里采用覆盖策略，假设只约束对上一个任务的参数
#         self.fisher = fisher_current
#         self.valid_keys = list(fisher_current.keys())
        
#         # 恢复训练模式
#         self.model.train()
#         print(f"==> [EWC] Fisher Matrix Updated. {len(self.fisher)} params protected.")
import torch
from .base import BaseCLAlgorithm
import numpy as np

class EWC(BaseCLAlgorithm):
    def __init__(self, model, optimizer, optimizer_center=None, ewc_lambda=5000, **kwargs):
        # [修正] 显式接收 optimizer_center，匹配 train_cl.py 的调用
        super().__init__(model, optimizer, optimizer_center=optimizer_center, **kwargs)
        self.ewc_lambda = ewc_lambda
        
        # 存储旧任务的 Fisher Matrix 和 参数
        self.fisher = {} 
        self.params = {} 
        
        # 只对 Backbone 做 EWC 保护
        self.valid_keys = [] 

    def observe(self, inputs, targets, task_loss_fn, scaler=None, **kwargs):
        """
        [关键修复] 
        1. 显式接收 scaler 参数，防止它混入 **kwargs 被传给 model 导致报错。
        2. 实现混合精度训练的梯度缩放逻辑。
        """
        self.optimizer.zero_grad()
        if hasattr(self, 'optimizer_center') and self.optimizer_center:
            self.optimizer_center.zero_grad()
        
        # 1. 计算当前任务 Loss
        # 此时 kwargs 已不包含 scaler，可以安全传给模型
        loss, score = task_loss_fn(inputs, targets, **kwargs)
        
        # 2. 添加 EWC 正则项
        if len(self.fisher) > 0:
            ewc_loss = 0
            # 遍历 backbone 参数 (保持你的逻辑：只保护 Backbone)
            for n, p in self.model.backbone.named_parameters():
                if n in self.fisher:
                    # 获取旧参数 (需确保在同一设备)
                    theta_old = self.params[n].to(p.device)
                    fisher_val = self.fisher[n].to(p.device)
                    
                    # Loss += (lambda/2) * Fisher * (theta - theta_old)^2
                    # 这里的 fisher 已经在 on_task_end 里做过平均
                    _loss = fisher_val * (p - theta_old).pow(2)
                    ewc_loss += _loss.sum()
            
            if np.random.rand() < 0.01: 
                print(f"[DEBUG] Task Loss: {loss.item():.4f} | EWC Raw: {ewc_loss.item():.6f} | Weighted EWC: {(self.ewc_lambda * ewc_loss).item():.4f}")
            loss += self.ewc_lambda * ewc_loss
            
        # 3. 反向传播 (适配 AMP Scaler)
        if scaler is not None:
            # [核心修复] FP16 下必须用 scaler 缩放梯度
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            
            if hasattr(self, 'optimizer_center') and self.optimizer_center:
                scaler.step(self.optimizer_center)
                
            scaler.update()
        else:
            # 普通 FP32 训练
            loss.backward()
            self.optimizer.step()
            if hasattr(self, 'optimizer_center') and self.optimizer_center:
                self.optimizer_center.step()
        
        return loss.item(), score

    def on_task_end(self, dataloader, task_loss_fn, **kwargs):
        """计算当前任务的 Fisher Matrix 并累积"""
        print("==> [EWC] Calculating Fisher Matrix for current task...")
        
        # 切换到 eval 模式以冻结 BN 统计量，但允许梯度反传
        self.model.train()
        
        fisher_current = {}
        self.params = {} # 清空并重新记录当前任务结束时的参数
        
        # 初始化 accumulator
        for n, p in self.model.backbone.named_parameters():
            if p.requires_grad:
                fisher_current[n] = torch.zeros_like(p.data)
                # 保存当前参数作为下次的 theta_old
                self.params[n] = p.data.clone()
        
        # 遍历数据计算梯度
        # 注意：计算 Fisher 不需要 Scaler，因为只看梯度大小不更新
        for i, batch in enumerate(dataloader):
            # 解包数据 (匹配 train_loader)
            inputs, targets, target_cam, target_view = batch
            
            inputs = inputs.cuda()
            targets = targets.cuda()
            target_cam = target_cam.cuda()
            target_view = target_view.cuda()
            
            self.model.zero_grad()
            
            # [核心保留] 手动前向传播，传入 TransReID 需要的额外参数
            outputs = self.model(inputs, cam_label=target_cam, view_label=target_view)
            
            # [核心保留] 计算 Loss (只用于求导)
            # 注意：这里的 task_loss_fn 是 processor.py 里的 val_loss_helper
            loss = task_loss_fn(outputs, targets, target_cam)
            loss.backward()
            
            # 累积 Backbone 梯度的平方
            for n, p in self.model.backbone.named_parameters():
                if n in fisher_current and p.grad is not None:
                    fisher_current[n] += p.grad.data.pow(2) / len(dataloader)
            
            if (i + 1) % 50 == 0:
                print(f"    EWC Fisher: Iter {i+1}/{len(dataloader)}")

        # 更新类属性
        self.fisher = fisher_current
        self.valid_keys = list(fisher_current.keys())
        
        all_fisher_values = torch.cat([f.view(-1) for f in self.fisher.values()])
        print(f"\\n{'='*10} Fisher Matrix Statistics {'='*10}")
        print(f"Mean Fisher: {all_fisher_values.mean().item():.8f}")
        print(f"Max  Fisher: {all_fisher_values.max().item():.8f}")
        print(f"{'='*40}\\n")
        
        print(f"==> [EWC] Fisher Matrix Updated. {len(self.fisher)} params protected (Backbone only).")