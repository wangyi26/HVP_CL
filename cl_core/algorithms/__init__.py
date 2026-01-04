from .finetuning import Finetuning
from .ewc import EWC

# 注册表
ALGORITHM_REGISTRY = {
    'finetuning': Finetuning,
    'naive': Finetuning,
    'ewc': EWC,
    #'er': ExperienceReplay,
    # 后续添加:
    # 'gdumb': GDumb,
    # 'inflora': Inflora
}

def build_algorithm(name, model, optimizer, **kwargs):
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(f"Algorithm '{name}' not found. Available: {list(ALGORITHM_REGISTRY.keys())}")
    
    return ALGORITHM_REGISTRY[name](model, optimizer, **kwargs)