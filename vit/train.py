import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler
from model.VIT import VisionTransformer
from model.CBAM import VisionTransformerCBAM
from model.PSCA import VisionTransformerPSCA
from model.CA import VisionTransformerCA
from model.SA import VisionTransformerSA
from model.LA import VisionTransformerLA
from model.FSA import VisionTransformerFSA
from model.GCSA import VisionTransformerGCSA
from utils.CustomDataset import CustomDataset
from utils.trainer import Trainer
from config.config import Config
import os
from datetime import datetime
import logging
from tqdm import tqdm


def setup_logging(log_dir):
    """配置训练日志"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'train_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def main():
    # 初始化配置和日志
    cfg = Config()
    logger = setup_logging(cfg.LOG_DIR)

    # 记录配置参数
    logger.info("Training Configuration:")
    for key, value in vars(cfg).items():
        if not key.startswith('__'):
            logger.info(f"{key}: {value}")

    # 加载数据集
    logger.info("Loading datasets...")
    dataset = CustomDataset(cfg.DATA_ROOT_TRAIN,cfg.DATA_ROOT_TEST, cfg.IMG_SIZE)
    train_loader, val_loader, _, n_classes = dataset.get_dataloaders(cfg.BATCH_SIZE)
    logger.info(f"Number of classes: {n_classes}")

    # 初始化模型
    logger.info("Initializing model...")
    model = VisionTransformerSA(
        img_size=cfg.IMG_SIZE,
        patch_size=cfg.PATCH_SIZE,
        in_channels=cfg.IN_CHANNELS,
        n_classes=n_classes,
        embed_dim=cfg.EMBED_DIM,
        depth=cfg.DEPTH,
        n_heads=cfg.N_HEADS,
        mlp_ratio=cfg.MLP_RATIO,
        dropout=cfg.DROPOUT
    ).to(cfg.DEVICE)
    print(model)

    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=1e-6)
    scaler = GradScaler()  # 混合精度训练

    # 初始化训练器
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, cfg.DEVICE)

    # 创建检查点目录
    os.makedirs(os.path.dirname(cfg.SAVE_PATH), exist_ok=True)

    best_val_acc = 0.0
    best_epoch = 0

    logger.info("Starting training...")
    for epoch in range(cfg.EPOCHS):
        logger.info(f"\nEpoch {epoch + 1}/{cfg.EPOCHS}")

        # 训练
        train_loss, train_acc = trainer.train_epoch(epoch, scaler)
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # 验证
        val_loss, val_acc = trainer.validate()
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Current LR: {current_lr:.6f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, cfg.SAVE_PATH)
            logger.info(f"New best model saved with val acc: {val_acc:.4f}")

    # 训练结束总结
    logger.info("\nTraining completed!")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")


if __name__ == '__main__':
    main()