import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()


        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.block(x)


class SP_UNet_Medium(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()

        # ---------- Spectral stem (wider, less bottleneck) ----------
        self.convsp1 = nn.Conv2d(in_ch, 24, 1, bias=True)
        self.bnsp1 = nn.BatchNorm2d(24)
        self.convsp2 = nn.Conv2d(24, 16, 1, bias=True)
        self.convsp3 = nn.Conv2d(16, 12, 1, bias=True)

        # ---------- Encoder ----------
        self.double1 = DoubleConv(12, 24)
        self.pool1 = nn.MaxPool2d(2)

        self.double2 = DoubleConv(24, 48)
        self.pool2 = nn.MaxPool2d(2)

        # ---------- Decoder ----------
        #self.up1 = nn.ConvTranspose2d(48, 48, kernel_size=2, stride=2)
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(48), nn.ReLU(inplace=False))
        self.double3 = DoubleConv(48, 24)

        #self.up2 = nn.ConvTranspose2d(24, 24, kernel_size=2, stride=2)
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(24, 24, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(24), nn.ReLU(inplace=False))
        self.double4 = DoubleConv(24 + 24, 12)

        # ---------- Final ----------
        self.final = nn.Conv2d(12, num_classes, kernel_size=1)

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):

        # spectral stem
        #x = self.relu(self.convsp1(x))
        x = self.relu(self.bnsp1(self.convsp1(x)))
        x = self.relu(self.convsp2(x))
        x = self.relu(self.convsp3(x))

        # encoder
        x1 = self.double1(x)
        x = self.pool1(x1)

        x = self.double2(x)
        x = self.pool2(x)

        # decoder
        x = self.up1(x)
        x = self.double3(x)

        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.double4(x)

        return self.final(x)