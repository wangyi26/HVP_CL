# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import collections.abc as container_abcs
from mmpretrain.models.backbones import VisionTransformer as mmvit
# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        qk = (q @ k.transpose(-2, -1)) * self.scale
        qk = qk.softmax(dim=-1)
        attn = self.attn_drop(qk)

        vv = ((v @ v.transpose(-2, -1)) * self.scale).softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, [qk, vv]

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads=num_heads,batch_first=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, key, value, return_attention=False):
        query = self.norm1(x)
        y, attn = self.attn(query, key, value)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=(224, 224), patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, num_heads_in_last_block=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, out_dims=768, pretrained=None, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.img_size = to_2tuple(img_size)
        self.patch_embed = PatchEmbed(
            img_size=self.img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth-1)]+[Block(
                dim=embed_dim, num_heads=num_heads_in_last_block, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[-1], norm_layer=norm_layer)])
        self.norm = norm_layer(embed_dim)
        scale = embed_dim ** -0.5 
        self.proj = nn.Parameter(scale * torch.randn(embed_dim, out_dims))
        self.apply(self._init_weights)
        if pretrained:
            self.init_weights_from_pretrained(pretrained)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights_from_pretrained(self, pretrained=None):
        checkpoint_model = torch.load(pretrained, map_location='cpu')
        if 'model' in checkpoint_model:
            param_dict = checkpoint_model['model']
        elif 'state_dict' in checkpoint_model:
            param_dict = checkpoint_model['state_dict']
        elif 'student' in checkpoint_model: ### for dino
            param_dict = checkpoint_model["student"]
        else:
            param_dict = checkpoint_model
        param_dict = {k.replace("backbone.", ""): v for k, v in param_dict.items()}
        param_dict = {k.replace("module.", ""): v for k, v in param_dict.items()}
        param_dict = {k.replace("student.", ""): v for k, v in param_dict.items()}
        # param_dict = {k.replace("layers.", "blocks."): v for k, v in param_dict.items()}
        count=0
        for k, v in param_dict.items():
            if k not in self.state_dict().keys():
                continue
            if 'head' in k or 'dist' in k or 'pre_logits' in k:
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                # To resize pos embedding when using model at different size from pretrained weights
                b, l, d = self.state_dict()[k].size()
                pos_emb = v[:,1:,:]
                pos_emb = F.interpolate(pos_emb.unsqueeze(0), size=(l-1,d),mode='bilinear')[0]
                v = torch.cat([v[:,0,:].unsqueeze(1), pos_emb], dim=1)
                param_dict[k] = v
            try:
                self.state_dict()[k].copy_(v)
                count +=1
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))
        print('Load %d / %d layers.'%(count,len(self.state_dict().keys())))
        msg = self.load_state_dict(param_dict, strict=False)
        print(msg)

    def interpolate_pos_encoding(self, x, h, w):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and self.img_size == (h,w):
            return self.pos_embed
        
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        OH = self.img_size[0] // self.patch_embed.patch_size
        OW = self.img_size[1] // self.patch_embed.patch_size
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, OH, OW, dim).permute(0, 3, 1, 2),
            scale_factor=(h0 / OH, w0 / OW),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-1] and int(h0) == patch_pos_embed.shape[-2]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        # print('x',x.shape)
        x = self.patch_embed(x)  # patch linear embedding
        # print('x1',x.shape)

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x):
        # print('begin:',x.shape)
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            x, _ = blk(x)
        x = self.norm(x)
        if self.proj is not None:
            x = x @ self.proj
        return x

def kd_vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        img_size=(256, 128),
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=6, num_heads_in_last_block=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        pretrained='/mnt/hdd3/wangxuanhan/denghuimin/DPALv2/work_dirs/lup1m_path-b_to_vit_tiny_from_global_local_diffusion_q-t_kv-s_predict-x0_concat_TandSloss/checkpoint0100.pth',
        # pretrained='/mnt/hdd3/wangxuanhan/research/vfm_research/learning_research/mae-dev/work_dirs/lupsub_kd_300e_vit_tiny/checkpoint.pth',
        # pretrained='/home/wangxuanhan/data/research/vfm_research/learning_research/saipv2/work_dirs/lup1m_path-b_to_vit_tiny_from_cls_patch_atten_moe_v8/checkpoint0100.pth',
        # pretrained='/mnt/hdd4/denghuimin/data/code/saipv2/work_dirs/proteus_PATH_training_8gpus_bz128/checkpoint_student.pth',
        # pretrained='/home/wangxuanhan/data/research/vfm_research/learning_research/saipv2/work_dirs/lup1m_path-l_to_vit_tiny_from_cls_patch_atten_moe_v2/checkpoint0100.pth',
        **kwargs)
    return model

def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3,num_heads_in_last_block=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        pretrained='pretrained/lupsub_csl_300e_vit_tiny.pth',
        # pretrained='/mnt/hdd3/wangxuanhan/research/vfm_research/learning_research/mae-dev/work_dirs/lup2m_csl_csmr_100e_ft_sep_vit_tiny/checkpoint.pth',
        # pretrained='/mnt/hdd3/wangxuanhan/research/vfm_research/downstream_tasks/reid/IRRA/pretrained_models/lupsub_hap_300e_vit_tiny.pth',
        **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        img_size=(256, 128),
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, num_heads_in_last_block=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        pretrained='/home/wangxuanhan/data/research/vfm_research/learning_research/saipv2/work_dirs/lup1m_path-b_to_vit_small_from_cls_patch_moe/checkpoint0100.pth',
        **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    # model = VisionTransformer(
    #     patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
    #     qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
    #     pretrained='/mnt/hdd3/wangxuanhan/research/vfm_research/learning_research/mae-dev/models/lup_hap_pre_vit_base.pth',
    #     **kwargs)
    
    model = VisionTransformer(
        img_size=(256, 128),
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, num_heads_in_last_block=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        pretrained='/mnt/hdd4/denghuimin/data/code/saipv2/work_dirs/lup1m_path-l_to_vit_base_from_cls_patch_atten_moe/checkpoint0060.pth',
        **kwargs)
    return model


class mm_vit(mmvit):
    """ Transformer-based Object Re-Identification
    """
    def __init__(self, embed_dim=768, out_dims=768, **args):
        super().__init__(**args)
        scale = embed_dim ** -0.5 
        self.proj = nn.Parameter(scale * torch.randn(embed_dim, out_dims))
        self.init_weights()

    def init_weights(self):

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            
            checkpoint_model = torch.load(self.init_cfg['checkpoint'], map_location='cpu')
            if 'model' in checkpoint_model:
                param_dict = checkpoint_model['model']
            elif 'state_dict' in checkpoint_model:
                param_dict = checkpoint_model['state_dict']
            elif 'student' in checkpoint_model: ### for dino
                print('load from student')
                param_dict = checkpoint_model["student"]
            else:
                param_dict = checkpoint_model
            param_dict = {k.replace("backbone_module.", ""): v for k, v in param_dict.items()}
            param_dict = {k.replace("module.", ""): v for k, v in param_dict.items()}
            # param_dict = {k.replace("blocks.", "layers."): v for k, v in param_dict.items()}
            filtered_params = {}
            for k, v in param_dict.items():
                
                new_k = k
                if 'decoder' in k or 'head' in k or 'cls' in k or 'mask' in k:
                    continue
                # if 'patch_emb' in k:
                #     new_k = k.replace("proj", "projection")
                if 'mlp' in k and k.index('mlp')!=0:
                    new_k = k.replace("mlp", "ffn")
                    fc_ind= int(new_k[new_k.index('fc')+2])
                    replaced_str = new_k[new_k.index('fc'):new_k.index('fc')+3]
                    
                    target_str = 'layers.'+str(fc_ind-1) if fc_ind>1 else 'layers.0.0'
                    new_k = new_k.replace(replaced_str, target_str)
                filtered_params[new_k] = v
            param_dict = filtered_params
            param_dict = {k.replace("blocks", "layers"): v for k, v in param_dict.items()}
            param_dict = {k.replace("norm", "ln"): v for k, v in param_dict.items()}
            if 'ln.weight' in param_dict.keys():
                param_dict['ln1.weight'] = param_dict['ln.weight']
                param_dict.pop('ln.weight')
            if 'ln.bias' in param_dict.keys():
                param_dict['ln1.bias'] = param_dict['ln.bias']
                param_dict.pop('ln.bias')
            
            msg = self.load_state_dict(param_dict, strict=False)
            print(msg)

    def forward(self, x):
        x = super().forward(x)[-1]
        if self.proj is not None:
            x = x @ self.proj
        return x.unsqueeze(1)
    
def sapiens_vit(**kwargs):
    
    model = mm_vit(
                arch='large',
                embed_dim=1024,
                img_size=(1024, 1024),
                patch_size=16,
                # qkv_bias=True,
                # drop_path_rate=0.3,
                with_cls_token=False,
                out_type='avg_featmap',
                # patch_cfg=dict(padding=2),
                # frozen_stages=12,
                init_cfg=dict(
                    type='Pretrained',
                    checkpoint='/home/wangxuanhan/data/research/vfm_research/learning_research/saipv2/pretrained_models/sapiens/sapiens_0.3b/sapiens_0.3b_epoch_1600_clean.pth'
                    ),
                **kwargs
            )
    return model

