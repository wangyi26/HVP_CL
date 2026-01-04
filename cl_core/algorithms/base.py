import torch

class BaseCLAlgorithm:
    def __init__(self, model, optimizer, **kwargs):
        """
        model: CLModelWrapper 实例
        optimizer: 优化器
        """
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def observe(self, inputs, targets, task_loss_fn, **kwargs):
        """
        处理一个 batch 的数据。
        必须返回 loss.item()
        """
        raise NotImplementedError

    def on_task_end(self, dataloader, task_loss_fn, **kwargs):
        """
        当前任务训练结束后的钩子函数。
        常用于 EWC 计算 Fisher Matrix，或 Herding 采样存 Buffer。
        """
        pass
    
    def on_epoch_end(self):
        """每个 Epoch 结束后的钩子 (可选)"""
        pass