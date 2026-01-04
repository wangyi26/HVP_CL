import torch
import torch.nn as nn
from external_models.transreid.make_model import weights_init_kaiming, weights_init_classifier
class BaseHead(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
    
    def forward(self, x):
        raise NotImplementedError

# 1. ReID / 检索任务头 (保留你现有的逻辑)
class ReIDHead(BaseHead):
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__(in_channels)
        self.classifier = nn.Linear(in_channels, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(in_channels)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.dropout = nn.Dropout(self.dropout_rate) if hasattr(self, 'dropout_rate') else nn.Dropout(0.0)

    def forward(self, x):
        feat = self.bottleneck(x)
        feat_cls = self.dropout(feat)

        return feat_cls, feat


class IdentityHead(BaseHead):
    """
    T2I 任务通常是对比学习，Head 逻辑集成在模型内部或 Loss 中。
    这里保留一个 IdentityHead 以防万一 wrapper 需要调用它。
    """
    def __init__(self, in_channels, **kwargs):
        super().__init__(in_channels)
        self.classifier = nn.Identity() # 占位

    def forward(self, x):
        return x, x
# 2. 属性识别任务头 (PAR) - 多标签分类
class AttributeHead(BaseHead):
    def __init__(self, in_channels, num_attributes, **kwargs):
        super().__init__(in_channels)
        self.classifier = nn.Linear(in_channels, num_attributes)
        # 属性识别通常不需要 BNNeck，直接用 BCE Loss

    def forward(self, x):
        return self.classifier(x), x

# 3. 姿态估计/关键点任务头 (HPE) - 像素回归 (Heatmaps)
# 假设输入特征是 Spatial 的 (B, C, H, W)
class PoseHead(BaseHead):
    def __init__(self, in_channels, num_joints, **kwargs):
        super().__init__(in_channels)
        # 简单的反卷积头用于上采样
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_joints, 1)
        )

    def forward(self, x):
        # 注意：这里需要 Backbone 返回空间特征 (feat_map)，不仅仅是 global_feat
        return self.deconv(x), x

# 4. 语义分割/解析任务头 (Parsing)
class ParsingHead(BaseHead):
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__(in_channels)
        self.segmenter = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, num_classes, 1)
        )
    
    def forward(self, x):
        # 输出 Logits mask
        return self.segmenter(x), x

# === 任务头工厂 ===
HEAD_REGISTRY = {
    'i2i': ReIDHead,      # 映射: i2i -> ReIDHead
    't2i': IdentityHead,
}

def  build_head(task_type, in_channels, **kwargs):
    if task_type not in HEAD_REGISTRY:
        raise ValueError(f"Unknown task type: {task_type}. Supported: {list(HEAD_REGISTRY.keys())}")
    return HEAD_REGISTRY[task_type](in_channels, **kwargs)