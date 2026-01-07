import torch
import torch.nn as nn
import copy
from .heads import build_head
# 引入刚才定义的 Adapter 和原来的 build_model
from external_models.irra.build import IRRAHeadAdapter, build_model as build_irra_impl

class UniversalCLModel(nn.Module):
    def __init__(self, visual_backbone, feature_dim, task_type='i2i'):
        super().__init__()
        
        self.backbone = visual_backbone 
        self.in_planes = feature_dim
        
        # === 核心组件容器 ===
        self.heads = nn.ModuleDict()     # 任务特定的 Heads
        self.stems = nn.ModuleDict()     # 任务特定的 Stems (Input Layers)
        
        # 记录每个任务的元数据
        self.task_types = {} 
        self.task_configs = {}           # 记录 config 以便处理不同的 img_size
        self.current_task = None
        
        # 初始化：检测并保存第一个任务(Task 1)的 Stem
        # 我们假设 visual_backbone 已经是初始化好的 Task 1 模型
        self._init_first_task_stem()

    def _init_first_task_stem(self):
        """
        将 Backbone 中现有的输入层组件提取出来，作为默认 Stem (用于 Task 1)
        """
        # ViT Stem 通常包含: patch_embed, cls_token, pos_embed
        # TransReID 还包含: sie_embed (Side Information Embedding), cam_num, view_num
        
        stem_state = {
            'patch_embed': self.backbone.patch_embed,
            'cls_token': self.backbone.cls_token,
            'pos_embed': self.backbone.pos_embed,
        }
        
        # 处理 TransReID 特有的属性
        if hasattr(self.backbone, 'sie_embed'):
            stem_state['sie_embed'] = self.backbone.sie_embed
        if hasattr(self.backbone, 'cam_num'):
            stem_state['cam_num'] = self.backbone.cam_num
        if hasattr(self.backbone, 'view_num'):
            stem_state['view_num'] = self.backbone.view_num
            
        # 封装成 ModuleDict 以便注册参数
        # 注意：普通属性(int)不能放进 ModuleDict，这里只存 nn.Module/Parameter
        self.stems['default'] = nn.ModuleDict({
            k: v for k, v in stem_state.items() if isinstance(v, (nn.Module, nn.Parameter))
        })
        # 额外存储非 Module 属性
        self.stems['default'].meta = {
            k: v for k, v in stem_state.items() if not isinstance(v, (nn.Module, nn.Parameter))
        }

    def add_task(self, task_name, num_classes=None, **kwargs):
        if task_name in self.heads: return

        current_task_type = kwargs.pop('task_type', 'i2i')
        cfg = kwargs.get('cfg', None) # 获取该任务的配置
        self.task_types[task_name] = current_task_type

        # =========================================================
        # 1. 创建任务特定的 Stem (Input Layer)
        # =========================================================
        # 策略：基于当前 Backbone 的 Stem 深度复制一份，然后根据新任务的配置进行调整 (如 Resize)
        
        # 深拷贝当前 Stem 作为基础
        new_stem = copy.deepcopy(self.stems['default']) 
        
        # 如果提供了 cfg，可能需要调整 PatchEmbed (分辨率变化)
        if cfg is not None:
            # 检查分辨率是否变化
            # 假设 patch_embed.img_size 是 tuple (H, W)
            current_h, current_w = new_stem['patch_embed'].img_size
            target_h, target_w = cfg.INPUT.SIZE_TRAIN
            
            if (current_h, current_w) != (target_h, target_w):
                #print(f"Adapting Stem for {task_name}: {current_h}x{current_w} -> {target_h}x{target_w}")
                
                # A. 重新初始化 PatchEmbed (参数会重置，或者你可以尝试插值权重)
                # 为了简单且有效，通常建议重新初始化或者加载针对该分辨率的预训练权重
                # 但在这里，我们可以利用 TransReID 的 PatchEmbed 类重新构建
                from external_models.transreid.backbones.vit_pytorch import PatchEmbed
                new_patch_embed = PatchEmbed(
                    img_size=(target_h, target_w),
                    patch_size=16, # 假设 patch size 不变
                    stride_size=16,
                    embed_dim=self.in_planes
                )
                new_stem['patch_embed'] = new_patch_embed
                
                # B. 插值 Positional Embedding
                # 利用 TransReID 中的 resize_pos_embed 工具函数
                from external_models.transreid.backbones.vit_pytorch import resize_pos_embed
                old_pos_embed = new_stem['pos_embed']
                # 注意：这里需要计算新的 grid size
                new_stem['pos_embed'] = nn.Parameter(
                    resize_pos_embed(old_pos_embed, old_pos_embed, target_h // 16, target_w // 16, hw_ratio=1)
                )

        # 注册 Stem
        self.stems[task_name] = new_stem

        # =========================================================
        # 2. 创建任务特定的 Head (Output Layer)
        # =========================================================
        if current_task_type == 't2i':
            # T2I Head: IRRA Adapter
            # 构建完整的 IRRA 模型
            full_irra_model = build_irra_impl(cfg, num_classes=num_classes)
            # 移除其 Visual Backbone (因为我们用 Shared Backbone)
            if hasattr(full_irra_model, 'vis_model'): 
                del full_irra_model.vis_model 
            
            self.heads[task_name] = IRRAHeadAdapter(full_irra_model)

        else:
            # I2I Head: ReIDHead (BNNeck + Classifier)
            self.heads[task_name] = build_head(
                'i2i',      
                self.in_planes,         
                num_classes=num_classes, 
                **kwargs
            )

    def set_current_task(self, task_name):
        """
        切换任务：
        1. 替换 Backbone 的 Input Stem 组件
        2. 冻结非当前任务的参数
        """
        self.current_task = task_name
        
        # 1. 获取当前任务的 Stem (如果是第一个任务，名字可能不在 keys 里，用 default)
        stem_key = task_name if task_name in self.stems else 'default'
        target_stem = self.stems[stem_key]
        
        # 2. [Monkey Patch] 替换 Backbone 组件
        self.backbone.patch_embed = target_stem['patch_embed']
        self.backbone.cls_token = target_stem['cls_token']
        self.backbone.pos_embed = target_stem['pos_embed']
        
        if 'sie_embed' in target_stem:
            self.backbone.sie_embed = target_stem['sie_embed']
        if hasattr(target_stem, 'meta'):
            self.backbone.cam_num = target_stem.meta.get('cam_num', 0)
            self.backbone.view_num = target_stem.meta.get('view_num', 0)
            
        # 3. 设置梯度状态 (Freeze/Unfreeze)
        self._set_grad_state(task_name)

    def _set_grad_state(self, current_task_name):
        # A. Shared Backbone (Body): 始终更新
        for param in self.backbone.parameters():
            param.requires_grad = True
            
        # B. Stems: 只更新当前任务
        for name, module in self.stems.items():
            requires_grad = (name == current_task_name) or (name == 'default' and current_task_name not in self.stems)
            for param in module.parameters():
                param.requires_grad = requires_grad
                
        # C. Heads: 只更新当前任务
        for name, module in self.heads.items():
            requires_grad = (name == current_task_name)
            for param in module.parameters():
                param.requires_grad = requires_grad

    def forward_backbone_seq(self, x):
        """
        执行 ViT Backbone 并强制返回序列特征 [B, N, C]
        我们手动重写 TransReID 的 forward流程，以确保拿到序列
        """
        # 1. Stem (此时已通过 set_current_task 替换为当前任务的 Stem)
        B = x.shape[0]
        x = self.backbone.patch_embed(x)
        
        cls_tokens = self.backbone.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Position Embedding & SIE (I2I 特有)
        if self.backbone.cam_num > 0 and self.backbone.view_num > 0:
             # 注意：这里需要外部传入 camera_id, view_id，但在 Shared Backbone 模式下
             # 如果是 T2I 任务，cam_num 通常设为 0，所以这里不会触发
             pass 
        
        x = x + self.backbone.pos_embed
        x = self.backbone.pos_drop(x)

        # 2. Body (Transformer Blocks)
        for blk in self.backbone.blocks:
            x = blk(x)
        
        x = self.backbone.norm(x)
        
        # 返回完整序列 [B, N, C]
        return x

    def forward(self, x, text=None, label=None, **kwargs):
        if self.current_task is None:
            # 默认行为：返回 pooled feature
            seq_feat = self.forward_backbone_seq(x)
            return seq_feat[:, 0]

        # 1. 获取共享特征 (序列)
        seq_feat = self.forward_backbone_seq(x)
        
        # 2. 进入 Head
        current_head = self.heads[self.current_task]
        task_type = self.task_types[self.current_task]

        if task_type == 't2i':
            # T2I Head (IRRA Adapter) 需要序列特征用于 MLM
            if self.training:
                batch = {'caption_ids': text, 'pids': label}
                batch.update(kwargs)
                return current_head(seq_feat, batch)
            else:
                # 推理
                if text is not None:
                    return current_head.model.encode_text(text)
                else:
                    return seq_feat[:, 0, :] # CLS token

        else:
            # I2I Head (BNNeck) 通常只需要 CLS token
            cls_feat = seq_feat[:, 0]
            feat_cls, feat_embed = current_head(cls_feat)
            
            if self.training:
                logits = current_head.classifier(feat_cls)
                # 返回符合 I2I 格式的 Tuple
                return logits, cls_feat, None 
            else:
                return feat_cls, None