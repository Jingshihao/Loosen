import torch
import torch.nn as nn
import torch.nn.functional as F

class LA(nn.Module):
    def __init__(self, num_heads=32):
        super(LA, self).__init__()
        self.num_heads = num_heads
        self.max = nn.MaxPool2d(2, stride=(2, 2))
        self.f1 = nn.Conv2d(num_heads, 1, 1, 1)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(1, num_heads, 1, 1)
        self.sig = nn.Sigmoid()
        self.v = nn.Conv2d(2, 1, 1, 1)

    def _process_patches(self, x):
        x_means = []
        x_maxs = []
        channel_chunks = torch.chunk(x, self.num_heads, dim=1)

        for p in range(self.num_heads):
            for i in range(3):
                for j in range(3):
                    x_patch = channel_chunks[p][:, :, 2 * i:2 * (i + 1), 2 * j:2 * (j + 1)]
                    x_mean = torch.mean(x_patch, dim=(1, 2, 3), keepdim=True)
                    x_max = self.max(x_patch)
                    x_max, _ = torch.max(x_max, dim=1, keepdim=True)
                    x_maxs.append(x_max)
                    x_means.append(x_mean)

        return x_means, x_maxs

    def _process_features(self, x_means, x_maxs):
        x_means = torch.stack(x_means, dim=1)
        x_maxs = torch.stack(x_maxs, dim=1)

        B = x_means.shape[0]
        x_means = x_means.reshape(B, self.num_heads, 3, 3)
        x_means = self.f2(self.relu(self.f1(x_means)))

        x_maxs = x_maxs.reshape(B, self.num_heads, 3, 3)
        x_maxs = self.f2(self.relu(self.f1(x_maxs)))

        return x_means, x_maxs

    def _fuse_features(self, x_means, x_maxs):
        x_means = torch.chunk(x_means, self.num_heads, dim=1)
        x_maxs = torch.chunk(x_maxs, self.num_heads, dim=1)

        x_fusions = []
        for mean, max_val in zip(x_means, x_maxs):
            x_fusion = self.v(torch.cat([mean, max_val], dim=1))
            x_fusions.append(x_fusion)

        x_fusion = torch.cat(x_fusions, dim=1)
        x_fusion = F.interpolate(x_fusion, (6, 6), mode='bilinear', align_corners=False)

        return torch.chunk(x_fusion, self.num_heads, dim=1)

    def forward(self, x):
        # Process patches
        x_means, x_maxs = self._process_patches(x)

        # Process features
        x_means, x_maxs = self._process_features(x_means, x_maxs)

        # Fuse features
        x_fusion = self._fuse_features(x_means, x_maxs)

        # Apply attention
        short_cut = torch.chunk(x, self.num_heads, dim=1)
        outputs = []
        for sc, fusion in zip(short_cut, x_fusion):
            outputs.append(sc * self.sig(fusion))

        out = torch.cat(outputs, dim=1)
        return out + x


# class LA(nn.Module):
#     def __init__(self, num_heads=32):
#         super(LA, self).__init__()
#         self.num_heads = num_heads
#         self.max = nn.MaxPool2d(2, stride=(2, 2))
#         self.f1 = nn.Conv2d(num_heads, 1, 1, 1)
#         self.relu = nn.ReLU()
#         self.f2 = nn.Conv2d(1, num_heads, 1, 1)
#         self.sig = nn.Sigmoid()
#         self.v = nn.Conv2d(2, 1, 1, 1)  # 保留（尽管输入维度可能需调整）
#
#     def _process_patches(self, x):
#         x_maxs = []
#         channel_chunks = torch.chunk(x, self.num_heads, dim=1)
#
#         for p in range(self.num_heads):
#             for i in range(3):
#                 for j in range(3):
#                     x_patch = channel_chunks[p][:, :, 2 * i:2 * (i + 1), 2 * j:2 * (j + 1)]
#                     x_max = self.max(x_patch)
#                     x_max, _ = torch.max(x_max, dim=1, keepdim=True)
#                     x_maxs.append(x_max)  # 仅保留mean
#
#         return x_maxs  # 原返回 (x_means, x_maxs)，现仅返回 x_means
#
#     def _process_features(self, x_maxs):
#         # 原代码中对 x_means 和 x_maxs 的相同处理，现仅用于 x_means
#         x_maxs = torch.stack(x_maxs, dim=1)  # 保持stack操作（原代码用torch.stack）
#
#         B = x_maxs.shape[0]
#         x_maxs = x_maxs.reshape(B, self.num_heads, 3, 3)
#         x_maxs = self.f2(self.relu(self.f1(x_maxs)))  # 与原始处理逻辑一致
#         x_maxs = F.interpolate(x_maxs, (6, 6), mode='bilinear', align_corners=False)
#
#         return x_maxs  # 原返回 (x_means, x_maxs)，现仅返回 x_means
#
#     def _fuse_features(self, x_maxs):
#         # 由于仅剩 x_means，需调整融合逻辑（原代码中 self.v 需要2通道输入）
#         # 此处假设直接返回 x_means 的分头结果（跳过融合步骤）
#         return torch.chunk(x_maxs, self.num_heads, dim=1)
#
#     def forward(self, x):
#         # Process patches (仅mean)
#         x_maxs = self._process_patches(x)
#
#         # Process features (仅mean)
#         x_maxs = self._process_features(x_maxs)
#
#         # Fuse features (直接分头)
#         x_fusion = self._fuse_features(x_maxs)
#
#         # Apply attention (与原逻辑一致)
#         short_cut = torch.chunk(x, self.num_heads, dim=1)
#         outputs = []
#         for sc, fusion in zip(short_cut, x_fusion):
#             outputs.append(sc * self.sig(fusion))
#
#         out = torch.cat(outputs, dim=1)
#         return out + x  # 残差连接保留





class AlexNet_LA(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet_LA, self).__init__()
        self.features = nn.Sequential(  # nn.Sequential能够将一系列层结构组合成一个新的结构   features用于提取图像特征的结构
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
            # input[3, 224, 224]  output[48, 55, 55] 48是卷积核个数 batch channel high weight
            # padding参数解释：如果是int型，比如说1 就是上下左右都补1列0  如果传入一个tuple(元组)的话 比如传入(1,2),1代表上下方各补一行0，2代表左右两侧各补两列0
            nn.ReLU(inplace=True),  # inplace这个参数可以理解为pytorch通过一种方法增加计算量，来降低内存使用容量的一种方法，可以通过这种方法在内存中载入更大的模型
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),  # output[128, 27, 27] 没有设置stride是因为这个卷积层的步长是1，而默认的步长就是1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 6, 6]
            LA(),
        )
        self.classifier = nn.Sequential(  # 包含了三层全连接层 是一个分类器
            nn.Dropout(p=0.5),  # p是每一层随机失活的比例  默认是0.5
            nn.Linear(128 * 6 * 6, 2048),  # 第一个全连接层，全连接层的节点个数是2048个
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),  # 第二个全连接层
            nn.ReLU(inplace=True),
              # 第三个全连接层 输入的是数据集 类别的个数，默认是1000

        )
        self.fc = nn.Linear(2048, num_classes)
        if init_weights:  # 初始化权重
            self._initialize_weights()

    def forward(self, x):  # 正向传播过程
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)  # 进行一个展平处理  将传进来的变量x进行展平，从index=1 这个维度开始 也就是channel
        x = self.classifier(x)
        return x  # 得到网络的预测输出

    def _initialize_weights(self):
        for m in self.modules():  # 遍历每一层结构
            if isinstance(m, nn.Conv2d):  # 判断是哪一个类别
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # kaiming_normal初始化权重方法
                if m.bias is not None:  # 如果偏置不为空的话，那么就用0去初始化
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):  # 如果这个实例是全连接层
                nn.init.normal_(m.weight, 0, 0.01)  # 采用normal，也就是正态分布来给我们的权重进行赋值，正态分布的均值等于0，方差是0.01
                nn.init.constant_(m.bias, 0)  # 将偏置初始化为0
