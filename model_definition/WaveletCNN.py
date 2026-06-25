import torch
import torch.nn as nn
from pytorch_wavelets import DWT1DForward


class WaveletCNN(nn.Module):
    """
    1-Level Daubechies-4 WaveletCNN.

    Hardcoded paper settings:
        levels = 1
        mother_wavelet = "db4"
        base_channels = 12
        dropout_rate = 0.1

    Kept as parameters:
        in_channels
        class_nums
    """

    def __init__(self, in_channels, class_nums):
        super().__init__()

        self.in_channels = in_channels
        self.class_nums = class_nums

        self.base_channels = 12
        self.dropout_rate = 0.1

        self.dwt_spectral = DWT1DForward(
            J=1,
            wave="db4",
            mode="zero"
        )

        with torch.no_grad():
            test_input = torch.zeros(1, 1, in_channels)
            low, high = self.dwt_spectral(test_input)
            low_channels = low.shape[-1]
            high_channels = high[0].shape[-1]

        projection_in_channels = in_channels + low_channels + high_channels

        self.projection = nn.Conv2d(
            projection_in_channels,
            self.base_channels,
            kernel_size=1
        )

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                self.base_channels,
                self.base_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(self.base_channels),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=self.dropout_rate),
        )

        self.seg_head = nn.Sequential(
            nn.Conv2d(
                self.base_channels,
                class_nums,
                kernel_size=1
            ),
            nn.BatchNorm2d(class_nums),
            nn.LeakyReLU(negative_slope=0.01)
        )

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(
                f"Expected input shape [B, C, H, W], got {tuple(x.shape)}"
            )

        B, C, H, W = x.shape

        if C != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {C}"
            )

        x_reshaped = x.permute(0, 2, 3, 1).reshape(-1, 1, C)

        low_spectral, high_spectral = self.dwt_spectral(x_reshaped)

        low_channels = low_spectral.shape[-1]
        high_channels = high_spectral[0].shape[-1]

        low_spectral = low_spectral.reshape(
            B, H, W, low_channels
        ).permute(0, 3, 1, 2)

        high_spectral = high_spectral[0].reshape(
            B, H, W, high_channels
        ).permute(0, 3, 1, 2)

        x = torch.cat(
            [x, low_spectral, high_spectral],
            dim=1
        )

        x = self.projection(x)
        x = self.conv_block(x)
        x = self.seg_head(x)

        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    net = WaveletCNN(in_channels=120, class_nums=3)

    input_tensor = torch.randn(1, 120, 14, 14)
    output = net(input_tensor)

    print("Input shape:", input_tensor.shape)
    print("Output shape:", output.shape)
    print("Trainable parameters:", count_parameters(net))