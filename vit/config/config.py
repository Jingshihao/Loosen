import torch

class Config:
    # 数据配置
    DATA_ROOT_TRAIN = 'C:\\Users\\Administrator\\Desktop\\38044\\select-first'
    DATA_ROOT_TEST = 'C:\\Users\\Administrator\\Desktop\\38044\\select-first'
    IMG_SIZE = 224  # 提高输入分辨率
    BATCH_SIZE = 128  # 根据GPU调整

    # 轻量级模型配置
    PATCH_SIZE = 16
    IN_CHANNELS = 3
    EMBED_DIM = 384  # 减小嵌入维度
    DEPTH = 6  # 减少Transformer层数
    N_HEADS = 6  # 减少注意力头数
    MLP_RATIO = 4
    DROPOUT = 0.1  # 增加dropout防过拟合

    # 训练优化
    EPOCHS = 300
    LR = 8e-4  # 更小的学习率
    WEIGHT_DECAY = 0.05  # 添加L2正则
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 保存与日志
    SAVE_PATH = './checkpoints/big/best_vit_model.pth'
    LOG_DIR = './logs/big/vit_base1'  # 添加训练日志目录