import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from timm.data.random_erasing import RandomErasing

# === 引入数据集类 ===
from .bases_i2i import ImageDataset
from .market1501 import Market1501
from .msmt17 import MSMT17


# === 引入采样器 (I2I 和 T2I 物理隔离) ===
from .samplers.sampler_i2i import RandomIdentitySampler as I2ISampler
from .samplers.sampler_i2i import RandomIdentitySampler_IdUniform as I2ISampler_IdUniform
from .samplers.sampler_t2i import RandomIdentitySampler as T2ISampler


# =====================================================================
# 1. 保留原项目 I2I 的 Collate Functions
# =====================================================================
def train_collate_fn(batch):
    """
    I2I Training Collate: 返回 (imgs, pids, camids, viewids)
    """
    imgs, pids, camids, viewids , _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids

def val_collate_fn(batch):
    """
    I2I Validation Collate: 额外返回 img_paths
    """
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids_batch, viewids, img_paths

# =====================================================================
# 2. 新增 T2I 的 Collate Function (IRRA 风格)
# =====================================================================
def collate_t2i(batch):
    batch_size = len(batch)
    elem = batch[0]
    batch_dict = {}
    for key in elem:
        if elem[key] is None: continue
        values = [d[key] for d in batch]
        if isinstance(elem[key], torch.Tensor):
            batch_dict[key] = torch.stack(values, 0)
        elif isinstance(elem[key], (int, float)):
             batch_dict[key] = torch.tensor(values)
        else:
            batch_dict[key] = values
    return batch_dict

# =====================================================================
# 3. 核心构建函数
# =====================================================================
def build_dataloader(cfg, dataset, task_type='i2i', is_train=True, dist=False):
    """
    统一 Loader 构建器。
    
    参数:
        cfg: 全局配置
        dataset: 已经实例化好的数据集对象 (包含 train, query, gallery 列表)
        task_type: 'i2i' 或 't2i'
        is_train: 是否为训练阶段
    """
    num_workers = cfg.DATALOADER.NUM_WORKERS

    # --- A. 构建 Transforms (完全保留原有逻辑) ---
    resize_t = T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3)
    train_transforms = T.Compose([
        resize_t,
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.PROB, mode='pixel', max_count=1, device='cpu'),
    ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    # =================================================================
    # 分支 1: I2I 任务 (TransReID / ResNet)
    # 逻辑严格遵循 make_dataloader.py
    # =================================================================
    if task_type == 'i2i':
        # --- 训练 Loader ---
        if is_train:
            # 1. 使用 ImageDataset 包装 (应用 Transforms)
            train_set = ImageDataset(dataset.train, train_transforms)
            
            # 2. 选择 Sampler
            # 兼容旧参数 IMS_PER_BATCH (Total Batch Size)
            batch_size = cfg.SOLVER.IMS_PER_BATCH if hasattr(cfg.SOLVER, 'IMS_PER_BATCH') else cfg.SOLVER.BATCH_SIZE
            
            if cfg.DATALOADER.SAMPLER == 'softmax_triplet':
                print('Using I2I Softmax Triplet Sampler')
                sampler = I2ISampler(dataset.train, batch_size, cfg.DATALOADER.NUM_INSTANCE)
                shuffle = False
                drop_last = True
            elif cfg.DATALOADER.SAMPLER == 'id_triplet':
                print('Using I2I ID Triplet Sampler')
                sampler = I2ISampler_IdUniform(dataset.train, cfg.DATALOADER.NUM_INSTANCE)
                shuffle = False
                drop_last = True
            else: # softmax
                print('Using I2I Softmax Sampler')
                sampler = None
                shuffle = True
                drop_last = True

            # 3. 构建 Loader
            train_loader = DataLoader(
                train_set, 
                batch_size=batch_size, 
                sampler=sampler, 
                shuffle=shuffle, 
                num_workers=num_workers, 
                collate_fn=train_collate_fn, # 原有 I2I Collate
                pin_memory=True,
                drop_last=drop_last
            )
            return train_loader

        # --- 验证 Loader ---
        else:
            # 合并 Query 和 Gallery 用于测试
            val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
            val_loader = DataLoader(
                val_set, 
                batch_size=cfg.TEST.BATCH_SIZE, 
                shuffle=False, 
                num_workers=num_workers,
                collate_fn=val_collate_fn # 原有 Val Collate
            )
            return val_loader

    # =================================================================
    # 分支 2: T2I 任务 (IRRA / CLIP)
    # =================================================================
    elif task_type == 't2i':
        # T2I 数据集通常在初始化时就需要 Transform，或者内部处理了
        # 这里假设 T2I dataset 对象 (如 CUHKPEDES) 有一个 transform 属性可以设置
        # 或者它本身已经是一个 PyTorch Dataset。
        # 为了简单，我们假设 dataset.train 已经是处理好的 Dataset
        
        # --- 训练 Loader ---
        if is_train:
            # IRRA 风格 Sampler
            batch_size = cfg.SOLVER.BATCH_SIZE
            
            if dist:
                sampler = T2ISamplerDDP(dataset.train, batch_size, cfg.DATALOADER.NUM_INSTANCE)
            else:
                sampler = T2ISampler(dataset.train, batch_size, cfg.DATALOADER.NUM_INSTANCE)
            
            train_loader = DataLoader(
                dataset.train,
                batch_size=batch_size,
                sampler=sampler,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_t2i, # T2I 专用 Collate
                pin_memory=True,
                drop_last=True
            )
            return train_loader
            
        # --- 验证 Loader ---
        else:
            val_loader = DataLoader(
                dataset.test,
                batch_size=cfg.TEST.BATCH_SIZE,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_t2i,
                pin_memory=True
            )
            return val_loader

    else:
        raise ValueError(f"Unknown task type: {task_type}")