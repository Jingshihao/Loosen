import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.VIT import VisionTransformer
from model.CBAM import VisionTransformerCBAM
from model.PSCA import VisionTransformerPSCA
from model.CA import VisionTransformerCA
from model.FSA import VisionTransformerFSA
from model.SA import VisionTransformerSA
from model.LA import VisionTransformerLA
from model.GCSA import VisionTransformerGCSA
from utils.CustomDataset import CustomDataset
from config.config import Config
import os
import logging
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def setup_logging(log_dir):
    """配置测试日志"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'test.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_pretrained_model(cfg, n_classes):
    """直接加载预训练模型权重"""
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

    # 直接加载模型权重
    if os.path.exists(cfg.SAVE_PATH):
        state_dict = torch.load(cfg.SAVE_PATH)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
            logging.info(f"Loaded model weights from {cfg.SAVE_PATH}")
            if 'val_acc' in state_dict:
                logging.info(f"Model's best validation accuracy: {state_dict['val_acc']:.4f}")
        else:
            # 如果保存的是模型直接的状态字典
            model.load_state_dict(state_dict)
            logging.info(f"Loaded raw model weights from {cfg.SAVE_PATH}")
    else:
        raise FileNotFoundError(f"No model weights found at {cfg.SAVE_PATH}")

    return model


def evaluate(model, test_loader, device):
    """评估模型性能"""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_labels = []
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy, np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(cm, class_names, save_path):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.close()


def save_classification_report(report, save_path):
    """保存分类报告"""
    with open(os.path.join(save_path, 'classification_report.txt'), 'w') as f:
        f.write(report)


def main():
    # 初始化配置和日志
    cfg = Config()
    logger = setup_logging(cfg.LOG_DIR)

    # 创建结果保存目录
    result_dir = os.path.join(cfg.LOG_DIR, 'test_results')
    os.makedirs(result_dir, exist_ok=True)

    # 加载数据集
    logger.info("Loading test dataset...")
    dataset = CustomDataset(cfg.DATA_ROOT_TRAIN, cfg.DATA_ROOT_TEST, cfg.IMG_SIZE)
    _, _, test_loader, n_classes = dataset.get_dataloaders(cfg.BATCH_SIZE)
    class_names = dataset.get_class_names()

    # 加载预训练模型
    logger.info("Loading pretrained model...")
    model = load_pretrained_model(cfg, n_classes)

    # 评估模型
    logger.info("Evaluating model on test set...")
    test_loss, test_acc, true_labels, pred_labels = evaluate(
        model, test_loader, cfg.DEVICE
    )

    # 打印基础指标
    logger.info(f"\nTest Results:")
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")

    # 生成详细报告
    cm = confusion_matrix(true_labels, pred_labels)
    report = classification_report(
        true_labels, pred_labels,
        target_names=class_names,
        digits=4
    )

    # 保存可视化结果
    plot_confusion_matrix(cm, class_names, result_dir)
    save_classification_report(report, result_dir)

    logger.info("\nClassification Report:")
    logger.info("\n" + report)
    logger.info(f"\nResults saved to {result_dir}")


if __name__ == '__main__':
    main()


