import torch
import torch.nn as nn
import torch.nn.functional as F





class CA(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x * self.sigmoid(out)








# 与AlexNet有两处不同： 1. 第一次的卷积核变小，步幅减小。 2. 第3，4，5层的卷积核数量增加了。
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
            CA(512),

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


def zfnet_CA(num_classes):
    model = ZFNet(num_classes=num_classes)
    return model
