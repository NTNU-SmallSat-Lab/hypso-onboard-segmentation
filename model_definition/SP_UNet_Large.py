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


class SP_UNet_Large(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()

        c1 = 16
        c2 = 32
        c3 = 64

        # ---------- Spectral stem ----------
        self.convsp1 = nn.Conv2d(in_ch, c2, kernel_size=1, bias=True)
        self.bnsp1 = nn.BatchNorm2d(c2)

        self.convsp2 = nn.Conv2d(c2, c1, kernel_size=1, bias=True)

        self.convsp3 = nn.Conv2d(c1, c1, kernel_size=1, bias=True)

        # ---------- Encoder ----------
        self.double1 = DoubleConv(c1, c2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.double2 = DoubleConv(c2, c3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---------- Decoder ----------
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c3, c3, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(c3), nn.ReLU(inplace=False))
        #self.up1 = nn.ConvTranspose2d(c3, c3, kernel_size=2, stride=2, bias=True)
        self.double3 = DoubleConv(c3, c2)

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c2, c2, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(c2), nn.ReLU(inplace=False))
        #self.up2 = nn.ConvTranspose2d(c2, c2, kernel_size=2, stride=2, bias=True)
        self.double4 = DoubleConv(c2 + c2, c1)

        # ---------- Final classifier ----------
        self.final = nn.Conv2d(c1, num_classes, kernel_size=1, bias=True)

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        # Spectral stem
        x = self.relu(self.bnsp1(self.convsp1(x)))
        x = self.relu(self.convsp2(x))
        x = self.relu(self.convsp3(x))

        # Encoder
        x1 = self.double1(x)
        x = self.pool1(x1)

        x = self.double2(x)
        x = self.pool2(x)

        # Decoder
        x = self.up1(x)
        x = self.double3(x)

        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)

        x = self.double4(x)

        # Logits
        return self.final(x)    