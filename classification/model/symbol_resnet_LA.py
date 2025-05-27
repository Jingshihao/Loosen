import torch
import torch.nn as nn
import torch.nn.functional as F

class LA(nn.Module):
    def __init__(self, num_heads=16):
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
            for i in range(7):
                for j in range(7):
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
        x_means = x_means.reshape(B, self.num_heads, 7, 7)
        x_means = self.f2(self.relu(self.f1(x_means)))

        x_maxs = x_maxs.reshape(B, self.num_heads, 7, 7)
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
        x_fusion = F.interpolate(x_fusion, (14, 14), mode='bilinear', align_corners=False)

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


class BasicBlock(nn.Module):
    """搭建BasicBlock模块"""
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        # 使用BN层是不需要使用bias的，bias最后会抵消掉
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)  # BN层, BN层放在conv层和relu层中间使用
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    # 前向传播
    def forward(self, X):
        identity = X
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))

        if self.downsample is not None:  # 保证原始输入X的size与主分支卷积后的输出size叠加时维度相同
            identity = self.downsample(X)

        return self.relu(Y + identity)


class BottleNeck(nn.Module):
    """搭建BottleNeck模块"""
    # BottleNeck模块最终输出out_channel是Residual模块输入in_channel的size的4倍(Residual模块输入为64)，shortcut分支in_channel
    # 为Residual的输入64，因此需要在shortcut分支上将Residual模块的in_channel扩张4倍，使之与原始输入图片X的size一致
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BottleNeck, self).__init__()

        # 默认原始输入为256，经过7x7层和3x3层之后BottleNeck的输入降至64
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)  # BN层, BN层放在conv层和relu层中间使用
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel, out_channel * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)  # Residual中第三层out_channel扩张到in_channel的4倍

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    # 前向传播
    def forward(self, X):
        identity = X

        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))

        if self.downsample is not None:  # 保证原始输入X的size与主分支卷积后的输出size叠加时维度相同
            identity = self.downsample(X)

        return self.relu(Y + identity)


class ResNet(nn.Module):
    """搭建ResNet-layer通用框架"""

    # num_classes是训练集的分类个数，include_top是在ResNet的基础上搭建更加复杂的网络时用到，此处用不到
    def __init__(self, residual, num_residuals, num_classes=1000, include_top=True):
        super(ResNet, self).__init__()

        self.out_channel = 64  # 输出通道数(即卷积核个数)，会生成与设定的输出通道数相同的卷积核个数
        self.include_top = include_top


        self.conv1 = nn.Conv2d(3, self.out_channel, kernel_size=7, stride=2, padding=3,
                               bias=False)  # 3表示输入特征图像的RGB通道数为3，即图片数据的输入通道为3
        self.bn1 = nn.BatchNorm2d(self.out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = self.residual_block(residual, 64, num_residuals[0])
        self.conv3 = self.residual_block(residual, 128, num_residuals[1], stride=2)
        self.conv4 = self.residual_block(residual, 256, num_residuals[2], stride=2)
        self.conv5 = self.residual_block(residual, 512, num_residuals[3], stride=2)
        self.model = LA()
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output_size = (1, 1)
            self.fc = nn.Linear(512 * residual.expansion, num_classes)

        # 对conv层进行初始化操作
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def residual_block(self, residual, channel, num_residuals, stride=1):
        downsample = None

        # 用在每个conv_x组块的第一层的shortcut分支上，此时上个conv_x输出out_channel与本conv_x所要求的输入in_channel通道数不同，
        # 所以用downsample调整进行升维，使输出out_channel调整到本conv_x后续处理所要求的维度。
        # 同时stride=2进行下采样减小尺寸size，(注：conv2时没有进行下采样，conv3-5进行下采样，size=56、28、14、7)。
        if stride != 1 or self.out_channel != channel * residual.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.out_channel, channel * residual.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * residual.expansion))

        block = []  # block列表保存某个conv_x组块里for循环生成的所有层
        # 添加每一个conv_x组块里的第一层，第一层决定此组块是否需要下采样(后续层不需要)
        block.append(residual(self.out_channel, channel, downsample=downsample, stride=stride))
        self.out_channel = channel * residual.expansion  # 输出通道out_channel扩张

        for _ in range(1, num_residuals):
            block.append(residual(self.out_channel, channel))

        # 非关键字参数的特征是一个星号*加上参数名，比如*number，定义后，number可以接收任意数量的参数，并将它们储存在一个tuple中
        return nn.Sequential(*block)

    # 前向传播
    def forward(self, X):
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.maxpool(Y)
        Y = self.conv2(Y)
        Y = self.conv3(Y)
        Y = self.conv4(Y)
        Y = self.model(Y)
        Y = self.conv5(Y)


        if self.include_top:
            Y = self.avgpool(Y)
            Y = torch.flatten(Y, 1)
            Y = self.fc(Y)

        return Y


# 构建ResNet-34模型
def resnet34_LA(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


# 构建ResNet-50模型
def resnet50(num_classes=1000, include_top=True):
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
# print(resnet34_LA())
