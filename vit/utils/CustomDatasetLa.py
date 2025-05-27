# # import os
# # import torch
# # from torchvision import transforms  # 导入transforms
# # from torch.utils.data import DataLoader  # 导入DataLoader
# # from torchvision.datasets import ImageFolder
# # from torch.utils.data import WeightedRandomSampler  # 可选，用于类别平衡
# #
# #
# # class CustomDataset:
# #     def __init__(self, root_dir_train, root_dir_test, img_size=224):
# #         self.root_dir_train = root_dir_train
# #         self.root_dir_test = root_dir_test
# #         self.img_size = img_size
# #         self.transform = transforms.Compose([
# #             transforms.Resize((img_size, img_size)),
# #             transforms.ToTensor(),
# #             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# #         ])
# #
# #     def get_dataloaders(self, batch_size=32):
# #         # 加载训练集
# #         train_dataset = ImageFolder(
# #             root=os.path.join(self.root_dir_train, 'train'),
# #             transform=self.transform
# #         )
# #
# #         # 加载验证集（尝试val或validation目录）
# #         val_dir = os.path.join(self.root_dir_train, 'val')
# #         if not os.path.exists(val_dir):
# #             val_dir = os.path.join(self.root_dir_train, 'validation')
# #
# #         val_dataset = ImageFolder(
# #             root=val_dir,
# #             transform=self.transform
# #         )
# #
# #         # 加载测试集
# #         test_dataset = ImageFolder(
# #             root=os.path.join(self.root_dir_test, 'test'),
# #             transform=self.transform
# #         )
# #
# #         # 创建数据加载器
# #         train_loader = DataLoader(
# #             train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
# #         )
# #         val_loader = DataLoader(
# #             val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
# #         )
# #         test_loader = DataLoader(
# #             test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
# #         )
# #
# #         # 获取类别数量（从训练集中获取）
# #         n_classes = len(train_dataset.classes)
# #
# #         # # 打印数据集信息
# #         # print(f"Dataset information:")
# #         # print(f"Number of classes: {n_classes}")
# #         # print(f"Class names: {train_dataset.classes}")
# #         # print(f"Training set size: {len(train_dataset)}")
# #         # print(f"Validation set size: {len(val_dataset)}")
# #         # print(f"Test set size: {len(test_dataset)}")
# #
# #         return train_loader, val_loader, test_loader, n_classes
#
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import WeightedRandomSampler
from timm.data import create_transform
from config.configLa import Config

cfg = Config()

class CustomDataset:
    def __init__(self, root_dir_train, root_dir_test, img_size=224):
        self.root_dir_train = root_dir_train
        self.root_dir_test = root_dir_test
        self.img_size = img_size
        self.transform = create_transform(
        input_size=cfg.IMG_SIZE,
        is_training=True,
        color_jitter=cfg.COLOR_JITTER,
        auto_augment=cfg.AA,
        interpolation='bicubic',
        re_prob=cfg.REPROB,
        re_mode=cfg.REMODE,
    )

        # 初始化时加载训练集以获取类别信息
        self.train_dataset = ImageFolder(
            root=os.path.join(self.root_dir_train, 'train'),
            transform=self.transform
        )

    def get_class_names(self):
        """返回训练数据集的类别名称列表"""
        return self.train_dataset.classes

    def get_dataloaders(self, batch_size=32):
        # 加载训练集（使用初始化时已加载的数据集）
        train_dataset = self.train_dataset

        # 加载验证集（尝试val或validation目录）
        val_dir = os.path.join(self.root_dir_train, 'val')
        if not os.path.exists(val_dir):
            val_dir = os.path.join(self.root_dir_train, 'validation')

        val_dataset = ImageFolder(
            root=val_dir,
            transform=self.transform
        )

        # 加载测试集
        test_dataset = ImageFolder(
            root=os.path.join(self.root_dir_test, 'test'),
            transform=self.transform
        )

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

        # 获取类别数量（从训练集中获取）
        n_classes = len(train_dataset.classes)

        return train_loader, val_loader, test_loader, n_classes


