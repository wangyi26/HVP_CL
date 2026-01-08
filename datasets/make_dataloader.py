import torch
from .i2i import make_dataloader as make_i2i_dataloader
from .t2i import build_dataloader as make_t2i_dataloader    
from utils.t2i_adapter import T2IArgsAdapter

def build_dataloader(cfg, dataset=None, task_type='i2i', is_train=True):
    if task_type == 'i2i':
        return make_i2i_dataloader(cfg)
    elif task_type == 't2i':
        args = T2IArgsAdapter(cfg, is_train)
        return make_t2i_dataloader(args)
    else:
        raise NotImplementedError