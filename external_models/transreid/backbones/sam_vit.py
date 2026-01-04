
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from collections import OrderedDict
from torch.nn import Module as BaseModule
from torch.nn import ModuleList
from .tinyvit import TinyViT

def trunc_normal_init(module: nn.Module,
                      mean: float = 0,
                      std: float = 1,
                      a: float = -2,
                      b: float = 2,
                      bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        #trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
        _no_grad_trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)  # type: ignore

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
def build_norm_layer(norm_cfg,embed_dims):
    assert norm_cfg['type'] == 'LN'
    norm_layer = nn.LayerNorm(embed_dims)
    return norm_cfg['type'],norm_layer
class MobileSam(BaseModule):

    def __init__(
        self,
        num_classes=1000,
        semantic_weight=0.,
        **kwargs
    ):
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()

        args = dict(
                in_chans=3,
                embed_dims=[64, 128, 160, 320],
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 5, 10],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.,
                drop_rate=0.,
                drop_path_rate=0.0,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
                layer_lr_decay=0.8,
                is_freeze = False
            )
        
        self.image_encoder = TinyViT(**args)
        self.num_features = args['embed_dims']
        self.num_features[-1] = 256
        for i in range(3):
            layer = build_norm_layer(dict(type='LN'), self.num_features[i])[1]
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)
        # semantic embedding
        # self.semantic_weight = semantic_weight
        # depths = args['depths']
        # if self.semantic_weight >= 0:
        #     self.semantic_embed_w = ModuleList()
        #     self.semantic_embed_b = ModuleList()
        #     for i in range(len(depths)):
        #         if i >= len(depths) - 1:
        #             i = len(depths) - 2
        #         semantic_embed_w = nn.Linear(2, self.num_features[i+1])
        #         semantic_embed_b = nn.Linear(2, self.num_features[i+1])
        #         for param in semantic_embed_w.parameters():
        #             param.requires_grad = False
        #         for param in semantic_embed_b.parameters():
        #             param.requires_grad = False
        #         trunc_normal_init(semantic_embed_w, std=.02, bias=0.)
        #         trunc_normal_init(semantic_embed_b, std=.02, bias=0.)
        #         self.semantic_embed_w.append(semantic_embed_w)
        #         self.semantic_embed_b.append(semantic_embed_b)
        #     self.softplus = nn.Softplus()

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        

    def init_weights(self, pretrained=None):
        logger = logging.getLogger("loading parameters.")
       
        ckpt = torch.load(pretrained,map_location='cpu')
        if 'teacher' in ckpt:
            ckpt = ckpt['teacher']

        if 'state_dict' in ckpt:
            _state_dict = ckpt['state_dict']
        elif 'model' in ckpt:
            _state_dict = ckpt['model']
        else:
            _state_dict = ckpt
        

        # state_dict = OrderedDict()
        # for k, v in _state_dict.items():
            
        #     if k.startswith('image_encoder.'):
        #         state_dict[k[14:]] = v

        # strip prefix of state_dict
        # if list(state_dict.keys())[0].startswith('module.'):
        #     state_dict = {k[7:]: v for k, v in state_dict.items()}

        res = self.load_state_dict(_state_dict, False)
        print('unloaded parameters:', res)

    def forward(
        self,
        batched_input: torch.Tensor
    ):
        
        image_embeddings = self.image_encoder(batched_input)
        # outs = []
        # for i, stage in enumerate(self.stages):
        #     x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
        #     if self.semantic_weight >= 0:
        #         sw = self.semantic_embed_w[i](semantic_weight).unsqueeze(1)
        #         sb = self.semantic_embed_b[i](semantic_weight).unsqueeze(1)
        #         x = x * self.softplus(sw) + sb
        #     if i in self.out_indices:
        #         norm_layer = getattr(self, f'norm{i}')
        #         out = norm_layer(out)
        #         out = out.view(-1, *out_hw_shape,
        #                        self.num_features[i]).permute(0, 3, 1,
        #                                                      2).contiguous()
        #         outs.append(out)
        outs = []
        for i, out in enumerate(image_embeddings):
            if i<3:
                norm_layer = getattr(self, f'norm{i}')
                b,c,h,w = out.size()
                out = out.view(b,c,-1).permute(0,2,1)
                out = norm_layer(out)
                out = out.view(-1, h, w, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        outs.append(image_embeddings[-1])
        x = self.avgpool(outs[-1])
        x = torch.flatten(x, 1)

        return x, outs
    
def mobilesam_tinyvit(**kwargs):
    model = MobileSam(**kwargs)
    return model


    
