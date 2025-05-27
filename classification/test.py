import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import logging
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from model.symbol_alexnet import AlexNet
from model.symbol_alexnet_CBAM import AlexNet_CBAM
from model.symbol_alexnet_CA import AlexNet_CA
from model.symbol_alexnet_FSA import AlexNet_FSA
from model.symbol_alexnet_GCSA import AlexNet_GCSA
from model.symbol_alexnet_SA import AlexNet_SA
from model.symbol_alexnet_LA import AlexNet_LA
from model.symbol_alexnet_PSCA import AlexNet_PSCA
from model.symbol_resnet_CBAM import resnet34_CBAM
from model.symbol_resnet import resnet34
from model.symbol_resnet_PSCA import resnet34_PSCA
from model.symbol_resnet_SA import resnet34_SA
from model.symbol_resnet_CA import resnet34_CA
from model.symbol_resnet_FSA import resnet34_FSA
from model.symbol_resnet_GCSA import resnet34_GCSA
from model.symbol_resnet_LA import resnet34_LA
from model.symbol_zfnet_CBAM import zfnet_CBAM
from model.symbol_zfnet_PSCA import zfnet_PSCA
from model.symbol_zfnet import zfnet
from model.symbol_zfnet_CA import zfnet_CA
from model.symbol_zfnet_FSA import zfnet_FSA
from model.symbol_zfnet_GCSA import zfnet_GCSA
from model.symbol_zfnet_SA import zfnet_SA
from model.symbol_zfnet_LA import zfnet_LA
from model.symbol_MobileNet import MobileNet
from model.symbol_MobileNet_SA import MobileNet_SA
from model.symbol_MobileNet_CA import MobileNet_CA
from model.symbol_MobileNet_FSA import MobileNet_FSA
from model.symbol_MobileNet_GCSA import MobileNet_GCSA
from model.symbol_MobileNet_CBAM import MobileNet_CBAM
from model.symbol_MobileNet_PSCA import MobileNet_PSCA
from model.symbol_MobileNet_LA import MobileNet_LA


# 配置类 - 集中管理所有参数
class TestConfig:
    # 设备配置
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 数据路径配置
    DATA_ROOT = r"C:\Users\Administrator\Desktop\38044\select-first"  # 数据集根目录
    LOG_DIR = r"C:\Users\Administrator\Desktop\classification\logs\ALEXNET\1"  # 日志保存目录

    # 模型配置
    MODEL_NAME = "MobileNet"  # 使用的模型名称
    SAVE_PATH = r"C:\Users\Administrator\Desktop\classification\weights\select\M\base"  # 模型权重保存路径

    # 数据加载配置
    IMG_SIZE = 224  # 输入图像大小
    BATCH_SIZE = 32  # 批次大小
    NUM_WORKERS = 4  # 数据加载工作线程数

    # 测试结果保存配置
    RESULT_DIR = os.path.join(LOG_DIR, "test_results")  # 测试结果保存目录


# 数据集类
class CustomDataset:
    def __init__(self, root_dir, img_size=224):
        self.root_dir = root_dir
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 加载测试集
        self.test_dataset = ImageFolder(
            root=os.path.join(self.root_dir, 'test'),
            transform=self.transform
        )

    def get_class_names(self):
        """返回类别名称列表"""
        return self.test_dataset.classes

    def get_dataloaders(self, batch_size=32):
        """获取数据加载器"""
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        n_classes = len(self.test_dataset.classes)
        return test_loader, n_classes


# 日志设置
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


# 模型加载
def load_pretrained_model(cfg, n_classes):
    """加载预训练模型"""
    # 根据配置选择模型
    model_classes = {
        'AlexNet': AlexNet,
        'AlexNet_CBAM': AlexNet_CBAM,
        'AlexNet_CA': AlexNet_CA,
        'AlexNet_FSA': AlexNet_FSA,
        'AlexNet_GCSA': AlexNet_GCSA,
        'AlexNet_SA': AlexNet_SA,
        'AlexNet_LA': AlexNet_LA,
        'AlexNet_PSCA': AlexNet_PSCA,
        'resnet34': resnet34,
        'resnet34_CBAM': resnet34_CBAM,
        'resnet34_PSCA': resnet34_PSCA,
        'resnet34_SA': resnet34_SA,
        'resnet34_CA': resnet34_CA,
        'resnet34_LA': resnet34_LA,
        'resnet34_FSA': resnet34_FSA,
        'resnet34_GCSA': resnet34_GCSA,
        'zfnet': zfnet,
        'zfnet_CBAM': zfnet_CBAM,
        'zfnet_PSCA': zfnet_PSCA,
        'zfnet_CA': zfnet_CA,
        'zfnet_FSA': zfnet_FSA,
        'zfnet_GCSA': zfnet_GCSA,
        'zfnet_SA': zfnet_SA,
        'zfnet_LA': zfnet_LA,
        'MobileNet': MobileNet,
        'MobileNet_CBAM': MobileNet_CBAM,
        'MobileNet_SA': MobileNet_SA,
        'MobileNet_CA': MobileNet_CA,
        'MobileNet_FSA': MobileNet_FSA,
        'MobileNet_GCSA': MobileNet_GCSA,
        'MobileNet_LA': MobileNet_LA,
        'MobileNet_PSCA': MobileNet_PSCA
    }
    if cfg.MODEL_NAME not in model_classes:
        raise ValueError(f"Unsupported model: {cfg.MODEL_NAME}")

    model = model_classes[cfg.MODEL_NAME](num_classes=n_classes).to(cfg.DEVICE)

    # 加载模型权重
    if os.path.exists(cfg.SAVE_PATH):
        state_dict = torch.load(cfg.SAVE_PATH)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
            logging.info(f"Loaded model weights from {cfg.SAVE_PATH}")
            if 'val_acc' in state_dict:
                logging.info(f"Model's best validation accuracy: {state_dict['val_acc']:.4f}")
        else:
            model.load_state_dict(state_dict)
            logging.info(f"Loaded raw model weights from {cfg.SAVE_PATH}")
    else:
        raise FileNotFoundError(f"No model weights found at {cfg.SAVE_PATH}")

    return model


# 评估函数
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


# 可视化工具
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


# 主函数
def main():
    # 初始化配置
    cfg = TestConfig()

    # 设置日志
    logger = setup_logging(cfg.LOG_DIR)

    # 创建结果目录
    os.makedirs(cfg.RESULT_DIR, exist_ok=True)

    # 加载数据集
    logger.info("Loading test dataset...")
    dataset = CustomDataset(cfg.DATA_ROOT, cfg.IMG_SIZE)
    test_loader, n_classes = dataset.get_dataloaders(cfg.BATCH_SIZE)
    class_names = dataset.get_class_names()

    # 加载模型
    logger.info("Loading pretrained model...")
    model = load_pretrained_model(cfg, n_classes)

    # 评估模型
    logger.info("Evaluating model on test set...")
    test_loss, test_acc, true_labels, pred_labels = evaluate(
        model, test_loader, cfg.DEVICE
    )

    # 输出结果
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

    # 保存结果
    plot_confusion_matrix(cm, class_names, cfg.RESULT_DIR)
    save_classification_report(report, cfg.RESULT_DIR)

    logger.info("\nClassification Report:")
    logger.info("\n" + report)
    logger.info(f"\nResults saved to {cfg.RESULT_DIR}")


if __name__ == '__main__':
    main()