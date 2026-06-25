import torch
import torch.nn as nn


class SpectralPointwise1D(nn.Module):
    def __init__(self, in_ch, ncls):
        super().__init__()

        self.convsp1 = nn.Conv2d(in_ch, 24, 1, bias=True)
        self.bnsp1 = nn.BatchNorm2d(24)

        self.convsp2 = nn.Conv2d(24, 12, 1, bias=True)
        self.convsp3 = nn.Conv2d(12, ncls, 1, bias=True)

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        if x.ndim != 2:
            raise RuntimeError(
                f"Expected 1D input with shape [B, C], got {tuple(x.shape)}"
            )

        x = x[:, :, None, None]  # [B, C] -> [B, C, 1, 1]

        x = self.relu(self.bnsp1(self.convsp1(x)))
        x = self.relu(self.convsp2(x))
        x = self.convsp3(x)

        x = x.squeeze(-1).squeeze(-1)  # [B, ncls, 1, 1] -> [B, ncls]

        return x