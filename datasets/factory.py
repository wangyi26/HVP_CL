import logging

# === I2I Dataset ===
from .i2i.market1501 import Market1501
from .i2i.msmt17 import MSMT17

# === T2I Datasets  ===
from .t2i.cuhkpedes import CUHKPEDES
from .t2i.icfgpedes import ICFGPEDES

def get_dataset(name, root, **kwargs):
    """
    统一数据集获取接口
    Args:
        name: 数据集名称 (e.g., 'Market1501', 'CUHK-PEDES')
        root: 数据根目录
    """
    # --- I2I Tasks ---
    if name == 'Market1501':
        return Market1501(root, **kwargs)
    if name == 'MSMT17':
        return MSMT17(root, **kwargs)
    
    # --- T2I Tasks ---
    if name == 'CUHK-PEDES':
        return CUHKPEDES(root, **kwargs)
    if name == 'ICFG-PEDES':
        return ICFGPEDES(root, **kwargs)
    
    raise KeyError(f"Unknown dataset: {name}")