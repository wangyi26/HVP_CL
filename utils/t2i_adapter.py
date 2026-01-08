# utils/t2i_adapter.py
import argparse

class T2IArgsAdapter:
    """
    [适配器] 将主项目的 cfg (YACS Node) 转换为 T2I 项目所需的 args (Namespace)。
    完全覆盖 get_args 函数中的参数定义。
    """
    def __init__(self, cfg, is_train=True):
        # ------------------------ General Settings ------------------------
        # 分布式训练通常由外部环境或主程序控制，这里从 cfg 读取或设为默认
        self.local_rank = getattr(cfg.MODEL, 'DEVICE_ID', 0) 
        self.name = getattr(cfg.T2I, 'NAME', 'baseline')
        self.output_dir = cfg.OUTPUT_DIR
        self.log_period = cfg.SOLVER.LOG_PERIOD
        self.eval_period = cfg.SOLVER.EVAL_PERIOD
        self.val_dataset = "test"  # 默认验证集使用测试集 split
        self.resume = getattr(cfg.T2I, 'RESUME', False)
        self.resume_ckpt_file = getattr(cfg.T2I, 'CKPT_FILE', "")

        # ------------------------ Model General Settings ------------------------
        self.pretrain_choice = getattr(cfg.T2I, 'PRETRAIN_CHOICE', 'ViT-B/16')
        self.temperature = getattr(cfg.T2I, 'TEMPERATURE', 0.02)
        
        # img_aug: 训练时默认开启，测试时关闭；也可以强制由 cfg 控制
        self.img_aug = is_train and getattr(cfg.T2I, 'IMG_AUG', True)
        self.backbone_type = getattr(cfg.T2I, 'BACKBONE_TYPE', 'vit_tiny')

        # ------------------------ Cross Modal Transformer ------------------------
        self.cmt_depth = getattr(cfg.T2I, 'CMT_DEPTH', 4)
        self.masked_token_rate = getattr(cfg.T2I, 'MASKED_TOKEN_RATE', 0.8)
        self.masked_token_unchanged_rate = getattr(cfg.T2I, 'MASKED_TOKEN_UNCHANGED_RATE', 0.1)
        self.lr_factor = getattr(cfg.T2I, 'LR_FACTOR', 5.0)
        
        # MLM: 自动判断或强制指定
        self.MLM = getattr(cfg.T2I, 'MLM', False)

        # ------------------------ Loss Settings ------------------------
        # 兼容 cfg.T2I.LOSS 为空的情况
        loss_cfg = getattr(cfg.T2I, 'LOSS', None)
        self.loss_names = getattr(loss_cfg, 'NAMES', 'sdm+id+mlm') if loss_cfg else 'sdm+id+mlm'
        self.mlm_loss_weight = getattr(loss_cfg, 'MLM_WEIGHT', 1.0) if loss_cfg else 1.0
        self.id_loss_weight = getattr(loss_cfg, 'ID_WEIGHT', 1.0) if loss_cfg else 1.0

        # ------------------------ Vision Transformer Settings ------------------------
        # cfg 中通常是 List，args 需要 Tuple
        img_size = getattr(cfg.T2I, 'IMG_SIZE', [384, 128])
        self.img_size = tuple(img_size)
        self.stride_size = getattr(cfg.T2I, 'STRIDE_SIZE', 16)

        # ------------------------ Text Transformer Settings ------------------------
        self.text_length = getattr(cfg.T2I, 'TEXT_LENGTH', 77)
        self.vocab_size = getattr(cfg.T2I, 'VOCAB_SIZE', 49408)

        # ------------------------ Solver ------------------------
        solver_cfg = getattr(cfg.T2I, 'SOLVER', None)
        self.optimizer = getattr(solver_cfg, 'OPTIMIZER', 'Adam') if solver_cfg else 'Adam'
        self.lr = getattr(solver_cfg, 'LR', 1e-5) if solver_cfg else 1e-5
        self.bias_lr_factor = getattr(solver_cfg, 'BIAS_LR_FACTOR', 2.0) if solver_cfg else 2.0
        self.momentum = getattr(solver_cfg, 'MOMENTUM', 0.9) if solver_cfg else 0.9
        self.weight_decay = getattr(solver_cfg, 'WEIGHT_DECAY', 4e-5) if solver_cfg else 4e-5
        self.weight_decay_bias = getattr(solver_cfg, 'WEIGHT_DECAY_BIAS', 0.0) if solver_cfg else 0.0
        self.alpha = getattr(solver_cfg, 'ALPHA', 0.9) if solver_cfg else 0.9
        self.beta = getattr(solver_cfg, 'BETA', 0.999) if solver_cfg else 0.999

        # ------------------------ Scheduler ------------------------
        sched_cfg = getattr(cfg.T2I, 'SCHEDULER', None)
        self.num_epoch = getattr(sched_cfg, 'NUM_EPOCH', 60) if sched_cfg else 60
        
        milestones = getattr(sched_cfg, 'MILESTONES', [20, 50]) if sched_cfg else [20, 50]
        self.milestones = tuple(milestones)
        
        self.gamma = getattr(sched_cfg, 'GAMMA', 0.1) if sched_cfg else 0.1
        self.warmup_factor = getattr(sched_cfg, 'WARMUP_FACTOR', 0.1) if sched_cfg else 0.1
        self.warmup_epochs = getattr(sched_cfg, 'WARMUP_EPOCHS', 5) if sched_cfg else 5
        self.warmup_method = getattr(sched_cfg, 'WARMUP_METHOD', "linear") if sched_cfg else "linear"
        self.lrscheduler = getattr(sched_cfg, 'LRSCHEDULER', "cosine") if sched_cfg else "cosine"
        self.target_lr = getattr(sched_cfg, 'TARGET_LR', 0.0) if sched_cfg else 0.0
        self.power = getattr(sched_cfg, 'POWER', 0.9) if sched_cfg else 0.9

        # ------------------------ Dataset ------------------------
        self.dataset_name = cfg.DATASETS.NAMES
        self.sampler = getattr(cfg.T2I, 'SAMPLER', 'random')
        self.num_instance = getattr(cfg.T2I, 'NUM_INSTANCE', 4)
        self.root_dir = cfg.DATASETS.ROOT_DIR
        self.batch_size = cfg.SOLVER.IMS_PER_BATCH
        self.test_batch_size = cfg.TEST.IMS_PER_BATCH
        self.num_workers = cfg.DATALOADER.NUM_WORKERS
        
        # training: 对应 parser 中的 --test dest='training' action='store_false'
        # 如果 is_train 为 True，则 training 为 True (除非 cfg 强制覆盖)
        self.training = is_train and getattr(cfg.T2I, 'TRAINING', True)
        
        self.data_ratio = getattr(cfg.T2I, 'DATA_RATIO', 1.0)
        
        # ------------------------ Distributed ------------------------
        # 补充：get_args 里没有显式定义 distributed，但 T2I 源码里经常用 args.distributed
        self.distributed = getattr(cfg.MODEL, 'DIST_TRAIN', False)

    def __repr__(self):
        # 方便打印调试
        return str(self.__dict__)