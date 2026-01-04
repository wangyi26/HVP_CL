import torch

def make_cl_optimizer(cfg, model, center_criterion=None):
    """
    统一优化器构建函数，自动识别 I2I (TransReID) 和 T2I (IRRA) 模型结构。
    """
    params = []
    
    # 基础参数
    base_lr = cfg.SOLVER.BASE_LR
    weight_decay = cfg.SOLVER.WEIGHT_DECAY
    bias_lr_factor = cfg.SOLVER.BIAS_LR_FACTOR
    weight_decay_bias = cfg.SOLVER.WEIGHT_DECAY_BIAS

    # 1. 定义通用参数过滤器
    def is_skipped(n):
        return any(k in n for k in ['bias', 'bn', 'norm', 'pos_embed', 'cls_token'])

    # ==========================================================
    # 策略 A: T2I 任务 (IRRA 结构)
    # 判断标准: model 拥有 text_backbone 属性 (由 ModelWrapper 注入)
    # ==========================================================
    if hasattr(model, 'text_backbone') and model.text_backbone is not None:
        # T2I 策略: Backbone (Visual+Text) 使用较小 LR，Head 使用 Base LR
        # 这里的 0.1 是 IRRA/CLIP 微调的经验值，你也可以在 cfg 里加一个参数控制
        backbone_scale = 0.1 
        
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            
            # 判断是否属于 Backbone (Visual 或 Text)
            # 我们的 Wrapper 把 IRRA 的 visual 映射为 backbone，text 映射为 text_backbone
            is_backbone = 'backbone' in n or 'text_backbone' in n or 'visual' in n
            
            lr = base_lr * backbone_scale if is_backbone else base_lr
            
            # Bias 特殊处理
            if "bias" in n:
                lr = lr * bias_lr_factor
                
            wd = 0.0 if is_skipped(n) else weight_decay
            
            params.append({'params': [p], 'lr': lr, 'weight_decay': wd})

    # ==========================================================
    # 策略 B: I2I 任务 (TransReID 结构)
    # ==========================================================
    else:
        # I2I 策略: 传统的 TransReID 优化器逻辑
        # 这里复刻了你 train_cl.py 里的简单逻辑，但加上了 TransReID 特有的处理
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            
            lr = base_lr
            wd = weight_decay
            
            # Bias 处理
            if "bias" in n:
                lr = base_lr * bias_lr_factor
                wd = weight_decay_bias
            
            # 如果不想 decay BN/LayerNorm
            if is_skipped(n):
                wd = 0.0
                
            params.append({'params': [p], 'lr': lr, 'weight_decay': wd})

    # ==========================================================
    # 构建 Optimizer
    # ==========================================================
    optimizer = torch.optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)

    # Center Loss (仅 I2I 有)
    optimizer_center = None
    if center_criterion is not None:
        optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

    return optimizer, optimizer_center