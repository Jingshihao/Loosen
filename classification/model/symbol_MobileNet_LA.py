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


class MobileNet_LA(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNet_LA, self).__init__()
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
            LA(),
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