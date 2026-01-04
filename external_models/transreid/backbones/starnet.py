"""
Implementation of Prof-of-Concept Network: StarNet.

We make StarNet as simple as possible [to show the key contribution of element-wise multiplication]:
    - like NO layer-scale in network design,
    - and NO EMA during training,
    - which would improve the performance further.

Created by: Xu Ma (Email: ma.xu1@northeastern.edu)
Modified Date: Mar/29/2024
"""
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from .vision_transformer import Block as ViTBlock

model_urls = {
    "starnet_s1": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s1.pth.tar",
    "starnet_s2": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s2.pth.tar",
    "starnet_s3": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s3.pth.tar",
    "starnet_s4": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s4.pth.tar",
}


class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


class StarNet(nn.Module):
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=-1, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = 32
        # stem layer
        self.stem = nn.Sequential(ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.ReLU6())
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] # stochastic depth
        # build stages
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            # down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1) if i_layer<len(depths)-1 else ConvBN(self.in_channel, embed_dim, 3, 1, 1)
            self.in_channel = embed_dim
            blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
        # head
        self.norm = nn.BatchNorm2d(self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.in_channel, num_classes) if num_classes>0 else nn.Identity()
        self.apply(self._init_weights)
        self.embed_dim = embed_dim
        self.num_features = [embed_dim]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        checkpoint_model = torch.load(pretrained, map_location='cpu')
        if 'model' in checkpoint_model:
            param_dict = checkpoint_model['model']
        elif 'state_dict' in checkpoint_model:
            param_dict = checkpoint_model['state_dict']
        elif 'student' in checkpoint_model: ### for dino
            print('load from student')
            param_dict = checkpoint_model["student"]
        else:
            param_dict = checkpoint_model
        param_dict = {k.replace("backbone.", ""): v for k, v in param_dict.items()}
        param_dict = {k.replace("module.", ""): v for k, v in param_dict.items()}
        param_dict = {k.replace("base.", ""): v for k, v in param_dict.items()}
        msg = self.load_state_dict(param_dict, strict=False)
        print(msg)

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        x = torch.flatten(self.avgpool(self.norm(x)), 1)
        return self.head(x), outs

class KDStarNet(StarNet):
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=-1, **kwargs):
        super().__init__(base_dim, depths, mlp_ratio, drop_path_rate, num_classes, **kwargs)
        self.norm = nn.BatchNorm1d(self.in_channel)
        self.neck = nn.Linear(self.embed_dim, 192)
        self.cls_token = nn.Parameter(torch.rand(1, 1, 192))
        self.vit_block = nn.ModuleList([ViTBlock(dim=192, num_heads=6, mlp_ratio=4, qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm),
                                        ViTBlock(dim=192, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm)])
        self.final_norm = nn.LayerNorm(192)
        self.apply(self._init_weights)
        self.embed_dim = 192
        self.num_features = [192]

    def forward_features(self, x):
        # print(x.size())
        x = self.stem(x)
        outs = []
        for stage in self.stages:
            x = stage(x)
            # print(x.size())
            outs.append(x)

        x = self.norm(x.flatten(2)).transpose(1, 2).contiguous()
        outs[-1] = self.neck(x)
        return outs
    
    def forward(self, x):
        outs = self.forward_features(x)
        b = outs[-1].shape[0]
        x = outs[-1]
        cls_token = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        for i, blk in enumerate(self.vit_block):
            x = blk(x)
        x = self.final_norm(x)
        return x[:, 0, :], outs
    
@register_model
def starnet_s1(pretrained=False, **kwargs):
    model = StarNet(24, [2, 2, 8, 3], **kwargs)
    if pretrained:
        url = model_urls['starnet_s1']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    return model


@register_model
def starnet_s2(pretrained=False, **kwargs):
    model = StarNet(32, [1, 2, 6, 2], **kwargs)
    if pretrained:
        url = model_urls['starnet_s2']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    return model


@register_model
def starnet_s3(pretrained=False, **kwargs):
    model = StarNet(32, [2, 2, 8, 4], **kwargs)
    if pretrained:
        url = model_urls['starnet_s3']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    return model

def kd_starnet_s3(pretrained=False, **kwargs):
    model = KDStarNet(32, [2, 2, 8, 4], **kwargs)
    return model

@register_model
def starnet_s4(pretrained=False, **kwargs):
    model = StarNet(32, [3, 3, 12, 5], **kwargs)
    if pretrained:
        url = model_urls['starnet_s4']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    return model


# very small networks #
@register_model
def starnet_s050(pretrained=False, **kwargs):
    return StarNet(16, [1, 1, 3, 1], 3, **kwargs)


@register_model
def starnet_s100(pretrained=False, **kwargs):
    return StarNet(20, [1, 2, 4, 1], 4, **kwargs)


@register_model
def starnet_s150(pretrained=False, **kwargs):
    return StarNet(24, [1, 2, 4, 2], 3, **kwargs)