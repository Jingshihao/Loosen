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

class CBAM(nn.Module):
    def __init__(self, c1,  ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        # 组合了通道注意力和空间注意力的CBAM模块
        # c1: 输入特征图的通道数，c2: 输出特征图的通道数，ratio: 通道注意力中的压缩比例，kernel_size: 空间注意力中的卷积核大小

        # 创建通道注意力模块
        self.channel_attention = ChannelAttention(c1, ratio)

        # 创建空间注意力模块
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # 首先应用通道注意力，然后应用空间注意力，得到最终的 CBAM 特征图
        out = self.channel_attention(x) * x  # 通过通道注意力权重缩放通道
        out = self.spatial_attention(out) * out  # 通过空间注意力权重缩放空间

        return out

class PatchEmbedding(nn.Module):
    """将图像分割为小块并嵌入"""

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # (batch_size, channels, height, width) -> (batch_size, n_patches, embed_dim)
        x = self.proj(x)  # (batch_size, embed_dim, n_patches^0.5, n_patches^0.5)
        x = x.flatten(2)  # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""

    def __init__(self, embed_dim=768, n_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        assert self.head_dim * n_heads == embed_dim, "Embedding dimension must be divisible by number of heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, n_patches, embed_dim = x.shape

        # 生成qkv
        qkv = self.qkv(x)  # (batch_size, n_patches, 3*embed_dim)
        qkv = qkv.reshape(batch_size, n_patches, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, n_heads, n_patches, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 应用注意力权重
        out = attn @ v  # (batch_size, n_heads, n_patches, head_dim)
        out = out.transpose(1, 2)  # (batch_size, n_patches, n_heads, head_dim)
        out = out.reshape(batch_size, n_patches, embed_dim)  # 合并多头

        # 线性投影
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MLP(nn.Module):
    """多层感知机"""

    def __init__(self, in_features, hidden_features, out_features, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer块"""

    def __init__(self, embed_dim=768, n_heads=12, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ca = CBAM(embed_dim)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            out_features=embed_dim,
            dropout=dropout
        )

    def forward(self, x):
        # 假设拼接后的 x 形状为 [batch_size, seq_len + 1, feature_dim]
        cls_token = x[:, 0, :]  # 提取 cls_token（第一个位置）
        cls_token = cls_token.unsqueeze(1)
        x_without_cls = x[:, 1:, :]  # 剩余部分（去除 cls_token）
        H = W = int(x_without_cls.shape[1] ** 0.5)
        B,_,C = x_without_cls.shape
        x_cbam = x_without_cls.view(B,H,W,C).permute(0,3,1,2)
        x_cbam = self.ca(x_cbam)
        x_cbam = x_cbam.view(B,C,-1).permute(0,2,1)
        x_cat = torch.cat((cls_token, x_cbam), dim=1)
        # 自注意力
        x = x + self.attn(self.norm1(x_cat))
        # x = x + x_cat
        # 前馈网络
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerCBAM(nn.Module):
    """完整的ViT模型"""

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_channels=3,
            n_classes=10,
            embed_dim=768,
            depth=12,
            n_heads=12,
            mlp_ratio=4,
            dropout=0.1
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer编码器
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

        # 初始化权重
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        batch_size = x.shape[0]

        # 生成patch嵌入
        x = self.patch_embed(x)  # (batch_size, n_patches, embed_dim)

        # 添加分类token
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # (batch_size, n_patches+1, embed_dim)

        # 添加位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 通过Transformer块
        for block in self.blocks:
            x = block(x)

        # 归一化
        x = self.norm(x)

        # 分类头
        cls_token_final = x[:, 0]  # 只取分类token
        x = self.head(cls_token_final)

        return x

