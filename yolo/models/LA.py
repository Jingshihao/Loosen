import torch
import torch.nn as nn
import torch.nn.functional as F


class LA(nn.Module):
    def __init__(self, in_c, out_c):
        super(LA, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x_2 = self.conv(x)
        B, C, H, W = x_2.shape  # 获取 x_2 的维度
        h = 2
        w = 2
        head = 8
        x = x_2.reshape(B, C, H // h, h, W // w, w).permute(0, 1, 3, 5, 2, 4).contiguous()
        N, C, reso_h, reso_w = x.shape
        x = x.view(N, C, reso_h * reso_w).permute(0, 2, 1)

        x = x.unsqueeze(3).permute(0, 3, 1, 2)
        x = x.reshape(N, reso_h * reso_w, C // head, head).permute(0, 3, 1, 2)

        x_mean = torch.mean(x, dim=(2, 3))
        x_mean = x_mean.reshape(B, head, reso_h, reso_w).permute(0, 1, 2, 3)
        x_mean1 = F.interpolate(x_mean, (H, W), mode='bilinear', align_corners=False)

        x_mean1 = x_mean1.unsqueeze(2)  # 添加一个维度以便广播
        shortcut = x_mean1 * x_2  # 广播相乘操作
        out = shortcut + x_2  # 直接相加
        out = self.conv(out)
        return out


if __name__ == '__main__':
    x = torch.randn(1, 128, 8, 8)
    model = LA(128, 128)
    output = model(x)
    print("LA:", output.shape)
