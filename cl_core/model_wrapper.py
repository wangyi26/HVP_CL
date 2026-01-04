import torch
import torch.nn as nn
from .heads import build_head

class UniversalCLModel(nn.Module):
    def __init__(self, original_model, feature_dim, task_type='i2i'):
        super().__init__()
        
        # 记录默认的任务类型 (初始化时的模型类型)
        self.task_type = task_type
        
        # 1. 识别模型架构
        # Case 1: IRRA (T2I) - 拥有 encode_text 方法
        if hasattr(original_model, 'encode_text'):
            self.backbone = original_model 
            self.text_backbone = original_model
            self.is_t2i = True
        # Case 2: TransReID (I2I)
        elif hasattr(original_model, 'base'):
            self.backbone = original_model.base
            self.text_backbone = None
            self.is_t2i = False
        else:
            self.backbone = original_model
            self.text_backbone = None
            self.is_t2i = False

        self.in_planes = feature_dim
        self.heads = nn.ModuleDict()
        self.current_task = None
        self.test_neck_feat = 'before' # 默认测试模式
        
        # [核心修复] 初始化任务类型字典
        self.task_types = {} 

    def add_task(self, task_name, num_classes=None, **kwargs):
        """
        添加新任务的 Head
        """
        if task_name in self.heads: 
            return

        # 1. 提取任务类型 (优先用传入的，否则用默认的)
        # [注意] 必须 pop 出来，防止传给 build_head 导致参数冲突
        current_task_type = kwargs.pop('task_type', self.task_type)
        
        # [核心修复] 记录该任务的类型，供 processor 使用
        self.task_types[task_name] = current_task_type

        # 2. 构建 Head
        # T2I 任务不需要分类头 (使用 Identity)
        if self.is_t2i or current_task_type == 't2i':
            self.heads[task_name] = nn.Identity()
        else:
            # I2I 任务：构建 ReIDHead (heads.py)
            # 此时 kwargs 已经很干净了，只包含 build_head 需要的参数 (如 neck)
            self.heads[task_name] = build_head(
                current_task_type,      # 'i2i'
                self.in_planes,         # feature dim
                num_classes=num_classes, 
                **kwargs
            )

    def set_current_task(self, task_name):
        self.current_task = task_name

    def forward(self, x, text=None, label=None, **kwargs):
        # ---------------------------------------------------
        # 分支 A: T2I 任务 (适配 IRRA 接口)
        # ---------------------------------------------------
        if self.is_t2i:
             if self.training:
                # 训练模式：打包参数传给 IRRA backbone
                batch_dict = {
                    'images': x,
                    'caption_ids': text,
                    'pids': label
                }
                batch_dict.update(kwargs) # 传入 mlm_ids 等
                return self.backbone(batch_dict)
             else:
                # 推理模式：分别提取特征
                if text is not None:
                    txt_feat = self.backbone.encode_text(text)
                    return txt_feat[torch.arange(txt_feat.shape[0]), text.argmax(dim=-1)]
                else:
                    img_feat = self.backbone.encode_image(x)
                    if len(img_feat.shape) == 3:
                        return img_feat[:, 0, :]
                    return img_feat

        # ---------------------------------------------------
        # 分支 B: I2I 任务 (适配 TransReID 接口)
        # ---------------------------------------------------
        else:
            # 1. Backbone 提取特征
            feat_tuple = self.backbone(x, **kwargs)
            
            if isinstance(feat_tuple, (tuple, list)):
                global_feat = feat_tuple[0]
                featmaps = feat_tuple[1] if len(feat_tuple) > 1 else None
            else:
                global_feat = feat_tuple
                featmaps = None
            
            if self.current_task is None: 
                return global_feat

            # 2. 进入 Head
            head = self.heads[self.current_task]
            feat_cls, feat_embed = head(global_feat)

            if self.training:
                # 训练返回：logits (通过 head.classifier 计算), global_feat, maps
                logits = head.classifier(feat_cls)
                return logits, global_feat, featmaps
            else:
                # 测试返回
                if hasattr(self, 'test_neck_feat') and self.test_neck_feat == 'after':
                    return feat_cls, featmaps
                return global_feat, featmaps