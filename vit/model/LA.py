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
        self.la = LA()
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
        x_la = x_without_cls.view(B,H,W,C).permute(0,3,1,2)
        x_la = self.la(x_la)
        x_la = x_la.view(B,C,-1).permute(0,2,1)
        x_cat = torch.cat((cls_token, x_la), dim=1)
        # 自注意力
        x = x + self.attn(self.norm1(x_cat))
        # x = x + x_cat
        # 前馈网络
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerLA(nn.Module):
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

