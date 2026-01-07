import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from timm.data.random_erasing import RandomErasing

# === 引入数据集类 ===
from .bases_i2i import ImageDataset
from .market1501 import Market1501
from .msmt17 import MSMT17
# [新增] 引入 T2I 数据集 Wrapper
from .bases_t2i import ImageTextDataset, ImageTextMLMDataset 

# === 引入采样器 ===
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
# 2. T2I Collate Function
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
    num_workers = cfg.DATALOADER.NUM_WORKERS

    # --- Transforms ---
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
    # 分支 1: I2I 任务
    # =================================================================
    if task_type == 'i2i':
        if is_train:
            train_set = ImageDataset(dataset.train, train_transforms)
            
            # 兼容旧参数
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
            else:
                print('Using I2I Softmax Sampler')
                sampler = None
                shuffle = True
                drop_last = True

            train_loader = DataLoader(
                train_set, 
                batch_size=batch_size, 
                sampler=sampler, 
                shuffle=shuffle, 
                num_workers=num_workers, 
                collate_fn=train_collate_fn,
                pin_memory=True,
                drop_last=drop_last
            )
            return train_loader

        else:
            val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
            val_loader = DataLoader(
                val_set, 
                batch_size=cfg.TEST.BATCH_SIZE, 
                shuffle=False, 
                num_workers=num_workers,
                collate_fn=val_collate_fn
            )
            return val_loader

    # =================================================================
    # 分支 2: T2I 任务 (核心修复)
    # =================================================================
    elif task_type == 't2i':
        if is_train:
            
            # 兼容：如果 cfg 中没有 text_length 定义，给个默认值
            text_len = getattr(cfg.MODEL, 'TEXT_LENGTH', 77)

            if 'mlm' in cfg.LOSS.NAME:
                train_set = ImageTextMLMDataset(dataset.train, train_transforms, text_length=text_len)
            else:
                train_set = ImageTextDataset(dataset.train, train_transforms, text_length=text_len)

            batch_size = cfg.SOLVER.BATCH_SIZE
            
            # T2I Sampler 需要传入原生列表 dataset.train (用于读取 PID)
            sampler_name = getattr(cfg.DATALOADER, 'SAMPLER', 'random')

            if sampler_name == 'random':
                # === 复现原日志的关键 ===
                # 使用标准的 RandomSampler (不进行 PK 采样，覆盖所有数据)
                print('Using T2I Random Sampler (All Data)')
                sampler = None # Loader 里设置 shuffle=True 即可
                shuffle_flag = True if sampler is None else False
                
            else:
                # === 默认行为 (Identity Sampler) ===
                # 这种采样对 ReID 性能通常更好，但会丢弃数据导致 iter 变少
                print('Using T2I Identity Sampler (PK Sampling)')
                sampler = T2ISampler(dataset.train, batch_size, cfg.DATALOADER.NUM_INSTANCE)
                shuffle_flag = False

            # 3. 构建 Loader
            train_loader = DataLoader(
                train_set,
                batch_size=batch_size,
                sampler=sampler,
                shuffle=shuffle_flag, # 如果 sampler 是 None (Random)，这里必须为 True
                num_workers=num_workers,
                collate_fn=collate_t2i,
                pin_memory=True,
                drop_last=True
            )
            return train_loader
            
        else:
            # 1. 图像 Loader (Gallery)
            num_workers = cfg.DATALOADER.NUM_WORKERS
            val_transforms = T.Compose([
                T.Resize(cfg.INPUT.SIZE_TEST),
                T.ToTensor(),
                T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
            ])

            # 1. 图像 Loader (Gallery)
            ds = dataset.test 
            val_img_set = ImageDataset(ds['image_pids'], ds['img_paths'], val_transforms)
            val_img_loader = DataLoader(
                val_img_set,
                batch_size=cfg.TEST.BATCH_SIZE,
                shuffle=False,
                num_workers=num_workers
            )
            
            # 2. 文本 Loader (Query)
            text_len = getattr(cfg.MODEL, 'TEXT_LENGTH', 77)
            val_txt_set = TextDataset(ds['caption_pids'], ds['captions'], text_length=text_len)
            val_txt_loader = DataLoader(
                val_txt_set,
                batch_size=cfg.TEST.BATCH_SIZE,
                shuffle=False,
                num_workers=num_workers
            )
            
            return val_img_loader, val_txt_loader 

    else:
        raise ValueError(f"Unknown task type: {task_type}")