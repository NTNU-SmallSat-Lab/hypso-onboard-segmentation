"""
Vitis AI QAT-ready model definitions.

Includes QuantStub/DeQuantStub wrappers, QAT-safe ReLU layers,
and a Cat wrapper for UNet skip connections.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional

import torch
import torch.nn as nn



try:
    import pytorch_nndct.nn as nndct_nn
    from pytorch_nndct.nn.modules import functional as nndct_functional

    QuantStub = nndct_nn.QuantStub
    DeQuantStub = nndct_nn.DeQuantStub
    NNDCT_AVAILABLE = True
except Exception:  
    from torch.ao.quantization import QuantStub, DeQuantStub

    nndct_functional = None
    NNDCT_AVAILABLE = False


class Cat(nn.Module):
    def __init__(self):
        super().__init__()
        self.cat = nndct_functional.Cat() if nndct_functional is not None else None

    def forward(self, tensors, dim: int = 0):
        if self.cat is not None:
            return self.cat(tensors, dim=dim)
        return torch.cat(tensors, dim=dim)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

#2D-Justo-UNet-Simple
class JustoUNetSimple(nn.Module):


    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()

        self.quant_stub = QuantStub()
        self.dequant_stub = DeQuantStub()

        # Encoder
        self.conv1 = nn.Conv2d(in_ch, 6, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(12)
        self.relu2 = nn.ReLU(inplace=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv3 = nn.Conv2d(12, 6, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(6)
        self.relu3 = nn.ReLU(inplace=False)

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv4 = nn.Conv2d(6, num_classes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant_stub(x)

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = self.up1(x)
        x = self.relu3(self.bn3(self.conv3(x)))

        x = self.up2(x)
        x = self.bn4(self.conv4(x))

        x = self.dequant_stub(x)
        return x


#1D-SpectralPointwise
class SpectralPointwise2D(nn.Module):

    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()

        self.quant_stub = QuantStub()
        self.dequant_stub = DeQuantStub()

        self.convsp1 = nn.Conv2d(in_ch, 24, kernel_size=1, bias=True)
        self.bnsp1 = nn.BatchNorm2d(24)
        self.relu1 = nn.ReLU(inplace=False)

        self.convsp2 = nn.Conv2d(24, 12, kernel_size=1, bias=True)
        self.relu2 = nn.ReLU(inplace=False)

        self.convsp3 = nn.Conv2d(12, num_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise RuntimeError(
                f"Expected 2D input with shape [B, C, H, W], got {tuple(x.shape)}"
            )

        x = self.quant_stub(x)

        x = self.relu1(self.bnsp1(self.convsp1(x)))
        x = self.relu2(self.convsp2(x))
        x = self.convsp3(x)

        x = self.dequant_stub(x)
        return x

#1D-SpectralPointwise
class SpectralPointwise1D(nn.Module):

    def __init__(self, in_ch: int, ncls: int):
        super().__init__()

        self.quant_stub = QuantStub()
        self.dequant_stub = DeQuantStub()

        self.convsp1 = nn.Conv2d(in_ch, 24, kernel_size=1, bias=True)
        self.bnsp1 = nn.BatchNorm2d(24)
        self.relu1 = nn.ReLU(inplace=False)

        self.convsp2 = nn.Conv2d(24, 12, kernel_size=1, bias=True)
        self.relu2 = nn.ReLU(inplace=False)

        self.convsp3 = nn.Conv2d(12, ncls, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise RuntimeError(
                f"Expected 1D input with shape [B, C], got {tuple(x.shape)}"
            )

        x = x[:, :, None, None]  # [B, C] -> [B, C, 1, 1]
        x = self.quant_stub(x)

        x = self.relu1(self.bnsp1(self.convsp1(x)))
        x = self.relu2(self.convsp2(x))
        x = self.convsp3(x)

        x = self.dequant_stub(x)
        x = x.squeeze(-1).squeeze(-1)  # [B, ncls, 1, 1] -> [B, ncls]
        return x

#2D-SpecPW-UNet-Large
class SPW_UNet_Large(nn.Module):
    """SP-UNet Large with Vitis AI QAT stubs."""

    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()

        self.quant_stub = QuantStub()
        self.dequant_stub = DeQuantStub()
        self.cat = Cat()

        c1 = 16
        c2 = 32
        c3 = 64

        # Spectral stem
        self.convsp1 = nn.Conv2d(in_ch, c2, kernel_size=1, bias=True)
        self.bnsp1 = nn.BatchNorm2d(c2)
        self.relu_sp1 = nn.ReLU(inplace=False)

        self.convsp2 = nn.Conv2d(c2, c1, kernel_size=1, bias=True)
        self.relu_sp2 = nn.ReLU(inplace=False)

        self.convsp3 = nn.Conv2d(c1, c1, kernel_size=1, bias=True)
        self.relu_sp3 = nn.ReLU(inplace=False)

        # Encoder
        self.double1 = DoubleConv(c1, c2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.double2 = DoubleConv(c2, c3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=False),
        )
        self.double3 = DoubleConv(c3, c2)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=False),
        )
        self.double4 = DoubleConv(c2 + c2, c1)

        self.final = nn.Conv2d(c1, num_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant_stub(x)

        # Spectral stem
        x = self.relu_sp1(self.bnsp1(self.convsp1(x)))
        x = self.relu_sp2(self.convsp2(x))
        x = self.relu_sp3(self.convsp3(x))

        # Encoder
        x1 = self.double1(x)
        x = self.pool1(x1)

        x = self.double2(x)
        x = self.pool2(x)

        # Decoder
        x = self.up1(x)
        x = self.double3(x)

        x = self.up2(x)
        x = self.cat([x, x1], dim=1)
        x = self.double4(x)

        x = self.final(x)
        x = self.dequant_stub(x)
        return x

#2D-SpecPW-UNet-Medium
class SPW_UNet_Medium(nn.Module):
    """SP-UNet Medium with Vitis AI QAT stubs."""

    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()

        self.quant_stub = QuantStub()
        self.dequant_stub = DeQuantStub()
        self.cat = Cat()

        # Spectral stem
        self.convsp1 = nn.Conv2d(in_ch, 24, kernel_size=1, bias=True)
        self.bnsp1 = nn.BatchNorm2d(24)
        self.relu_sp1 = nn.ReLU(inplace=False)

        self.convsp2 = nn.Conv2d(24, 16, kernel_size=1, bias=True)
        self.relu_sp2 = nn.ReLU(inplace=False)

        self.convsp3 = nn.Conv2d(16, 12, kernel_size=1, bias=True)
        self.relu_sp3 = nn.ReLU(inplace=False)

        # Encoder
        self.double1 = DoubleConv(12, 24)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.double2 = DoubleConv(24, 48)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=False),
        )
        self.double3 = DoubleConv(48, 24)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(24, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=False),
        )
        self.double4 = DoubleConv(24 + 24, 12)

        self.final = nn.Conv2d(12, num_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant_stub(x)

        # Spectral stem
        x = self.relu_sp1(self.bnsp1(self.convsp1(x)))
        x = self.relu_sp2(self.convsp2(x))
        x = self.relu_sp3(self.convsp3(x))

        # Encoder
        x1 = self.double1(x)
        x = self.pool1(x1)

        x = self.double2(x)
        x = self.pool2(x)

        # Decoder
        x = self.up1(x)
        x = self.double3(x)

        x = self.up2(x)
        x = self.cat([x, x1], dim=1)
        x = self.double4(x)

        x = self.final(x)
        x = self.dequant_stub(x)
        return x

#2D-SpecPW-UNet-Small
class SPW_UNet_Small(nn.Module):
    """SP-UNet Small with Vitis AI QAT stubs."""

    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()

        self.quant_stub = QuantStub()
        self.dequant_stub = DeQuantStub()
        self.cat = Cat()

        # Spectral stem
        self.convsp1 = nn.Conv2d(in_ch, 24, kernel_size=1, bias=True)
        self.bnsp1 = nn.BatchNorm2d(24)
        self.relu_sp1 = nn.ReLU(inplace=False)

        self.convsp2 = nn.Conv2d(24, 12, kernel_size=1, bias=True)
        self.relu_sp2 = nn.ReLU(inplace=False)

        self.convsp3 = nn.Conv2d(12, 12, kernel_size=1, bias=True)
        self.relu_sp3 = nn.ReLU(inplace=False)

        # Encoder
        self.double1 = DoubleConv(12, 12)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.double2 = DoubleConv(12, 24)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(24, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=False),
        )
        self.double3 = DoubleConv(24, 12)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(12, 12, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=False),
        )
        self.double4 = DoubleConv(12 + 12, 8)

        self.final = nn.Conv2d(8, num_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant_stub(x)

        # Spectral stem
        x = self.relu_sp1(self.bnsp1(self.convsp1(x)))
        x = self.relu_sp2(self.convsp2(x))
        x = self.relu_sp3(self.convsp3(x))

        # Encoder
        x1 = self.double1(x)
        x = self.pool1(x1)

        x = self.double2(x)
        x = self.pool2(x)

        # Decoder
        x = self.up1(x)
        x = self.double3(x)

        x = self.up2(x)
        x = self.cat([x, x1], dim=1)
        x = self.double4(x)

        x = self.final(x)
        x = self.dequant_stub(x)
        return x


MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    "justounetsimple": JustoUNetSimple,
    "JustoUNetSimple": JustoUNetSimple,

    "spectralpointwise2d": SpectralPointwise2D,
    "SpectralPointwise2D": SpectralPointwise2D,

    # Optional backward-compatible aliases
    "spectralpointwise1D": SpectralPointwise2D,
    "SpectralPointwise1D": SpectralPointwise2D,

    "spw_unet_large": SPW_UNet_Large,
    "SPW_UNet_Large": SPW_UNet_Large,
    "spw_unet_medium": SPW_UNet_Medium,
    "SPW_UNet_Medium": SPW_UNet_Medium,
    "spw_unet_small": SPW_UNet_Small,
    "SPW_UNet_Small": SPW_UNet_Small,
}


def build_model(model_name: str, in_ch: int, num_classes: int) -> nn.Module:
    try:
        cls = MODEL_REGISTRY[model_name]
    except KeyError as exc:
        valid = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model_name={model_name!r}. Valid names: {valid}"
        ) from exc

    return cls(in_ch=in_ch, num_classes=num_classes)


def _extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    """Accept common checkpoint formats and return a plain state_dict."""

    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value
        return checkpoint

    raise TypeError(
        "checkpoint must be a state_dict mapping or a checkpoint containing "
        "one of: state_dict, model_state_dict, model, net"
    )


def _strip_prefix_if_present(state_dict: Mapping[str, torch.Tensor], prefix: str) -> OrderedDict:
    keys = list(state_dict.keys())
    if keys and all(key.startswith(prefix) for key in keys):
        return OrderedDict((key[len(prefix):], value) for key, value in state_dict.items())
    return OrderedDict(state_dict.items())


def load_fp32_weights_for_qat(
    model: nn.Module,
    checkpoint_or_path: Any,
    *,
    strict: bool = False,
    map_location: str | torch.device = "cpu",
):
    """Load FP32 weights into the QAT-stubbed model.

    strict=False is the practical default because QuantStub/DeQuantStub and
    Vitis/QAT wrappers can add keys that are not present in the original FP32
    checkpoint. The returned object is PyTorch's IncompatibleKeys result.
    """

    if isinstance(checkpoint_or_path, (str, Path)):
        checkpoint = torch.load(checkpoint_or_path, map_location=map_location)
    else:
        checkpoint = checkpoint_or_path

    state_dict = _extract_state_dict(checkpoint)
    state_dict = _strip_prefix_if_present(state_dict, "module.")

    return model.load_state_dict(state_dict, strict=strict)


__all__ = [
    "NNDCT_AVAILABLE",
    "Cat",
    "DoubleConv",
    "JustoUNetSimple",
    "SpectralPointwise1D",
    "SpectralPointwise2D",
    "SPW_UNet_Large",
    "SPW_UNet_Medium",
    "SPW_UNet_Small",
    "MODEL_REGISTRY",
    "build_model",
    "load_fp32_weights_for_qat",
]

