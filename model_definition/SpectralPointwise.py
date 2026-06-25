import torch
import torch.nn as nn


class SpectralPointwise(nn.Module):
    def __init__(self, in_ch, ncls):
        super().__init__()

        self.convsp1 = nn.Conv2d(in_ch, 24, 1, bias=True)
        self.bnsp1 = nn.BatchNorm2d(24)

        self.convsp2 = nn.Conv2d(24, 12, 1, bias=True)
        self.convsp3 = nn.Conv2d(12, ncls, 1, bias=True)

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.relu(self.bnsp1(self.convsp1(x)))  # only BN here
        x = self.relu(self.convsp2(x))
        x = self.convsp3(x)
        return x