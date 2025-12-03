import torch
import torch.nn as nn
import torch.nn.functional as F


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x_pooled = self.pool(x)
        return x, x_pooled


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class CFDSurrogateUNet(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32, out_channels: int = 3):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.down1 = DownBlock(in_channels, c1)
        self.down2 = DownBlock(c1, c2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up1 = UpBlock(c3, c2)
        self.up2 = UpBlock(c2, c1)

        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=1)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cl_head = nn.Linear(c3, 1)
        self.cd_head = nn.Linear(c3, 1)

    def forward(self, x):
        s1, x1 = self.down1(x)
        s2, x2 = self.down2(x1)
        bottleneck = self.bottleneck(x2)

        x = self.up1(bottleneck, s2)
        x = self.up2(x, s1)
        fields_pred = self.out_conv(x)

        pooled = self.gap(bottleneck).flatten(1)
        cl_pred = self.cl_head(pooled)
        cd_pred = self.cd_head(pooled)
        return fields_pred, cl_pred, cd_pred
