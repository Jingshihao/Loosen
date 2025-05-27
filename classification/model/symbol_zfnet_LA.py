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
            for i in range(9):
                for j in range(9):
                    x_patch = channel_chunks[p][:, :, 3 * i:3 * (i + 1), 3 * j:3 * (j + 1)]
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
        x_means = x_means.reshape(B, self.num_heads, 9, 9)
        x_means = self.f2(self.relu(self.f1(x_means)))

        x_maxs = x_maxs.reshape(B, self.num_heads, 9, 9)
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
        x_fusion = F.interpolate(x_fusion, (27, 27), mode='bilinear', align_corners=False)

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
            LA(),

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


def zfnet_LA(num_classes):
    model = ZFNet(num_classes=num_classes)
    return model
