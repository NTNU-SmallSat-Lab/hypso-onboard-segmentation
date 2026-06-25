import torch
import torch.nn as nn


from model_architecture.SpectralPointwise import SpectralPointwise
from model_architecture.SpectralPointwise1D import SpectralPointwise1D
from model_architecture.SP_UNet_Large import SP_UNet_Large
from model_architecture.SP_UNet_Medium import SP_UNet_Medium
from model_architecture.SP_UNet_Small import SP_UNet_Small

from model_architecture.JustoLiuNet import JustoLiuNet
from model_architecture.JustoLiuNet_BN import JustoLiuNet_BN
from model_architecture.JustoUNetSimple import JustoUNetSimple

from model_architecture.WaveletCNN import WaveletCNN

MODEL_REGISTRY = {
    "spectralpointwise": SpectralPointwise,
    "spectralpointwise1d": SpectralPointwise1D,
    "sp_unet_large": SP_UNet_Large,
    "sp_unet_medium": SP_UNet_Medium,
    "sp_unet_small": SP_UNet_Small,
    "justoliunet": JustoLiuNet,
    "justoliunet_bn": JustoLiuNet_BN,
    "justounetsimple": JustoUNetSimple,
    "waveletcnn": WaveletCNN,
}


def build_model(name_model: str, in_channels: int, n_classes: int) -> nn.Module:
    name_model = str(name_model).lower()

    if name_model not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model name: {name_model}. "
            f"Available models are: {available}"
        )

    model_cls = MODEL_REGISTRY[name_model]
    return model_cls(in_channels, n_classes)
