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



class MobileNet_PSCA(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNet_PSCA, self).__init__()
        self.nclass = num_classes

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            PSCA(512),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, self.nclass)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x