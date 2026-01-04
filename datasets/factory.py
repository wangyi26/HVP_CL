# HVP_CL/datasets/factory.py
from .market1501 import Market1501
from .msmt17 import MSMT17
from .cuhkpedes import CUHKPEDES
from .icfgpedes import ICFGPEDES
# ... import 其他数据集

def get_dataset(name, root, **kwargs):
    """统一数据集获取接口"""
    # I2I Tasks
    if name == 'Market1501': return Market1501(root, **kwargs)
    if name == 'MSMT17': return MSMT17(root, **kwargs)
    
    # T2I Tasks
    if name == 'CUHK-PEDES': return CUHKPEDES(root, **kwargs)
    if name == 'ICFG-PEDES': return ICFGPEDES(root, **kwargs)
    
    raise KeyError(f"Unknown dataset: {name}")