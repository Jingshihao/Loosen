import torch
import torch.nn as nn
import torch.nn.functional as F





class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # 通道注意力模块，用于增强通道间的特征关系
        # in_planes: 输入特征图的通道数，ratio: 通道压缩比例
        # 自适应平均池化和自适应最大池化，用于捕获全局通道信息
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 第一个卷积层，用于通道压缩
        self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()

        # 第二个卷积层，用于通道恢复
        self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        # Sigmoid 激活函数，将通道注意力权重缩放到 [0, 1] 范围内
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化和最大池化后，通过两个卷积层进行通道注意力计算
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))

        # 将平均池化和最大池化的结果相加，并通过 Sigmoid 缩放得到最终的通道注意力权重
        out = self.sigmoid(avg_out + max_out)

        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        # 空间注意力模块，用于增强特征图的空间关系
        # kernel_size: 空间注意力操作的卷积核大小，padding 根据 kernel_size 自动确定
        # 计算平均值和最大值，并进行通道融合
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        # Sigmoid 激活函数，将空间注意力权重缩放到 [0, 1] 范围内
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 计算特征图的平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # 将平均值和最大值在通道维度上拼接，用于进行空间注意力操作
        x = torch.cat([avg_out, max_out], dim=1)

        # 通过卷积操作并通过 Sigmoid 缩放得到最终的空间注意力权重
        x = self.conv(x)

        return self.sigmoid(x)

class PSCA(nn.Module):
    def __init__(self, c1, ratio=16, kernel_size=7):
        super(PSCA, self).__init__()
        # 组合了通道注意力和空间注意力的CBAM模块
        # c1: 输入特征图的通道数，c2: 输出特征图的通道数，ratio: 通道注意力中的压缩比例，kernel_size: 空间注意力中的卷积核大小

        # 创建通道注意力模块
        self.channel_attention = ChannelAttention(c1, ratio)

        # 创建空间注意力模块
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self,x):
        out1 = self.channel_attention(x) * x
        out2 = self.spatial_attention(x) * x
        out = out1+out2
        return  out




class ZFNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(ZFNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=2),  # input[3, 224, 224]  output[96, 111, 111]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[96, 55, 55]

            nn.Conv2d(96, 256, kernel_size=5, padding=2),  # output[256, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[256, 27, 27]

            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # output[512, 27, 27]
            nn.ReLU(inplace=True),
            PSCA(512),

            nn.Conv2d(512, 1024, kernel_size=3, padding=1),  # output[1024, 27, 27]
            nn.ReLU(inplace=True),

            nn.Conv2d(1024, 512, kernel_size=3, padding=1),  # output[512, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[512, 13, 13]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512 * 13 * 13, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),


        )
        self.fc = nn.Linear(4096, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def zfnet_PSCA(num_classes):
    model = ZFNet(num_classes=num_classes)
    return model
