import torch


class Config:
    # ========== 数据配置 ==========
    DATA_ROOT_TRAIN = 'C:\\Users\\Administrator\\Desktop\\38044\\select-first'
    DATA_ROOT_TEST = 'C:\\Users\\Administrator\\Desktop\\38044\\select-first'
    IMG_SIZE = 224
    BATCH_SIZE = 64  # 减小batch size提升训练稳定性

    # ========== 模型结构（固定不变） ==========
    PATCH_SIZE = 16
    IN_CHANNELS = 3
    EMBED_DIM = 384
    DEPTH = 6
    N_HEADS = 6
    MLP_RATIO = 4
    DROPOUT = 0.3  # 增加防过拟合
    DROP_PATH = 0.2  # 新增Stochastic Depth

    # ========== 训练超参数 ==========
    EPOCHS = 300  # 延长训练
    LR = 0.0001  # 初始学习率
    MIN_LR = 1e-8  # 最小学习率
    WEIGHT_DECAY = 0.05  # L2正则
    WARMUP_EPOCHS = 20  # 学习率预热
    OPTIMIZER = 'adamw'  # adamw或lion
    SCHEDULER = 'cosine'  # 学习率调度策略

    # ========== 数据增强 ==========
    COLOR_JITTER = 0.4  # 颜色抖动强度
    AA = 'rand-m9-mstd0.5-inc1'  # AutoAugment
    MIXUP = 0.8  # MixUp概率
    CUTMIX = 1.0  # CutMix概率
    REPROB = 0.25  # Random Erasing概率
    REMODE = 'pixel'  # 随机擦除模式

    # ========== 硬件与日志 ==========
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SAVE_PATH = './checkpoints/best_model.pth'
    LOG_DIR = './logs/training_logs'
    USE_AMP = True  # 启用混合精度
    GRAD_CLIP = 1.0  # 梯度裁剪阈值
    EMA_DECAY = 0.9999  # EMA衰减率