from .base import BaseCLAlgorithm
import torch.cuda.amp as amp

class Finetuning(BaseCLAlgorithm):
    def observe(self, inputs, targets, task_loss_fn, scaler=None, **kwargs):
        self.optimizer.zero_grad()
        if hasattr(self, 'optimizer_center') and self.optimizer_center:
            self.optimizer_center.zero_grad()
            
     
        loss, score = task_loss_fn(inputs, targets, **kwargs)
        
        if scaler is not None:
            # 如果有 scaler，使用混合精度反向传播
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            
            if hasattr(self, 'optimizer_center') and self.optimizer_center:
                for param in self.optimizer_center.param_groups:
                    param['lr'] = self.optimizer.param_groups[0]['lr']
                scaler.step(self.optimizer_center)
                
            scaler.update()
        else:
            # 标准 FP32 反向传播
            loss.backward()
            self.optimizer.step()
            if hasattr(self, 'optimizer_center') and self.optimizer_center:
                for param in self.optimizer_center.param_groups:
                    param['lr'] = self.optimizer.param_groups[0]['lr']
                self.optimizer_center.step()
        
        return loss.item(), score