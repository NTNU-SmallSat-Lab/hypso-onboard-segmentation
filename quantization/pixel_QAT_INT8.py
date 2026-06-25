"""
Expected files/folders:

    Script dependency:
        - models_vitis_ai_qat.py
        Must be in the same folder as this script

    Dataset:
        - datasets/<dataset_name>/data0.npy
        - datasets/<dataset_name>/label0.npy
    One data/label pair is expected for each image ID from 0 to --n-images - 1.

    FP32 checkpoints:
        - weights_P32/<model_name>/best_model.pth

    Checkpoint override:
        - --fp32-checkpoint /path/to/checkpoint.pth
    Use this when running one specific model.

    Outputs:
        - out_folder/<dataset_name>_P<patch_size>/<model_name>/
    Contains QAT checkpoints, qat_summary.json, and deployable exports.

    XModel export:
        - out_folder/<dataset_name>_P<patch_size>/<model_name>/deployable/*.xmodel
    Created when --export-xmodel is enabled.

    Runtime:
        - Run inside a Vitis AI PyTorch environment with pytorch_nndct installed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


try:
    import pytorch_nndct  
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from models_vitis_ai_qat import build_model


POINTWISE_MODELS = [
    "spectralpointwise2d",
    "spectralpointwise1D",
    "SpectralPointwise2D",
    "SpectralPointwise1D",
]

MODEL_CHECKPOINT_ALIASES = {
    "spectralpointwise2d": [
        "spectralpointwise1D",
        "spectralpointwise",
    ],
}


DEFAULT_VAL_IDS = [0, 1, 5, 35, 37, 41, 47, 52]
DEFAULT_TEST_IDS = [12, 16, 17, 23, 24, 46, 54, 58]


@dataclass
class Metrics:
    acc: float
    miou: float
    iou: List[float]
    loss: Optional[float] = None


@dataclass
class RunSummary:
    model_name: str
    fp32_checkpoint: str
    output_dir: str
    best_qat_checkpoint: str
    latest_qat_checkpoint: str
    deployable_dir: Optional[str]
    xmodel_files: List[str]
    fp32_val: Metrics
    best_val: Metrics
    best_epoch: int
    final_val: Metrics
    final_test: Optional[Metrics]
    effective_pixels_per_step: int


def parse_id_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_fixed_split(n_images: int, val_ids: Sequence[int], test_ids: Sequence[int]):
    all_ids = list(range(n_images))
    val_ids, test_ids = list(val_ids), list(test_ids)
    val_set, test_set, all_set = set(val_ids), set(test_ids), set(all_ids)

    if len(val_ids) != len(val_set):
        raise ValueError("Duplicate IDs found in val_ids")
    if len(test_ids) != len(test_set):
        raise ValueError("Duplicate IDs found in test_ids")
    if val_set & test_set:
        raise ValueError(f"Validation and test IDs overlap: {sorted(val_set & test_set)}")
    if (val_set | test_set) - all_set:
        raise ValueError(f"IDs outside range 0..{n_images - 1}: {sorted((val_set | test_set) - all_set)}")

    train_ids = [i for i in all_ids if i not in val_set and i not in test_set]
    if sorted(train_ids + val_ids + test_ids) != all_ids:
        raise RuntimeError("Internal split validation failed")
    return all_ids, train_ids, val_ids, test_ids


def squeeze_label(label: np.ndarray) -> np.ndarray:
    label = np.asarray(label)
    if label.ndim == 3 and 1 in (label.shape[0], label.shape[-1]):
        label = np.squeeze(label)
    if label.ndim != 2:
        raise ValueError(f"Expected label shape [H,W], got {label.shape}")
    return label


def as_chw(data: np.ndarray, label: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    label = squeeze_label(label)
    h, w = label.shape

    if data.ndim == 2:
        if data.shape != (h, w):
            raise ValueError(f"Data shape {data.shape} does not match label shape {label.shape}")
        return data[None]
    if data.ndim != 3:
        raise ValueError(f"Expected data [C,H,W] or [H,W,C], got {data.shape}")
    if data.shape[1] == h and data.shape[2] == w:
        return data
    if data.shape[0] == h and data.shape[1] == w:
        return np.transpose(data, (2, 0, 1))
    raise ValueError(f"Could not infer channel axis for data {data.shape} and label {label.shape}")


def load_data_cache(base: Path, ids: Sequence[int], mmap_mode: Optional[str]):
    data_cache, label_cache = {}, {}
    print(f"[cache] Opening {len(ids)} image/label pairs from {base} (mmap_mode={mmap_mode})")
    for i in ids:
        data_path = base / f"data{i}.npy"
        label_path = base / f"label{i}.npy"
        if not data_path.exists():
            raise FileNotFoundError(data_path)
        if not label_path.exists():
            raise FileNotFoundError(label_path)

        data = np.load(data_path, mmap_mode=mmap_mode)
        label = squeeze_label(np.load(label_path, mmap_mode=mmap_mode))
        data_cache[i], label_cache[i] = data, label
    return data_cache, label_cache


def build_chw_views(data_cache, label_cache, ids: Sequence[int]) -> Dict[int, np.ndarray]:
    return {i: as_chw(data_cache[i], label_cache[i]) for i in ids}


def infer_in_channels(chw_cache, image_id: int) -> int:
    return int(chw_cache[image_id].shape[0])


def valid_label(v: int, num_classes: int, ignore_index: Optional[int]) -> bool:
    if ignore_index is not None and v == ignore_index:
        return False
    return 0 <= v < num_classes


class ClassPixelIndex:
    def __init__(self, label_cache, ids: Sequence[int], num_classes: int, ignore_index: Optional[int]):
        self.num_classes = num_classes
        self.by_class: Dict[int, List[Tuple[int, np.ndarray, np.ndarray]]] = {c: [] for c in range(num_classes)}
        self.present_classes: List[int] = []

        for image_id in ids:
            y = squeeze_label(label_cache[image_id])
            for cls in range(num_classes):
                if ignore_index is not None and cls == ignore_index:
                    continue
                ys, xs = np.nonzero(y == cls)
                if ys.size:
                    self.by_class[cls].append((image_id, ys.astype(np.int32), xs.astype(np.int32)))

        self.present_classes = [cls for cls, items in self.by_class.items() if items]
        if not self.present_classes:
            raise ValueError("Balanced sampling requested, but no valid class coordinates were found")

        counts = {cls: int(sum(len(ys) for _, ys, _ in items)) for cls, items in self.by_class.items() if items}
        print(f"[sampler] Balanced class pixel counts: {counts}")

    def sample_one(self, rng: np.random.Generator) -> Tuple[int, int, int]:
        cls = int(rng.choice(self.present_classes))
        choices = self.by_class[cls]
        image_id, ys, xs = choices[int(rng.integers(0, len(choices)))]
        j = int(rng.integers(0, len(ys)))
        return image_id, int(ys[j]), int(xs[j])


class RandomPixel1x1Dataset(Dataset):
    """Samples independent spectral pixels as literal 1x1 Conv2d inputs.
    Each item is x=[C, 1, 1], y=[1, 1].  A DataLoader batch therefore becomes
    [B, C, 1, 1], which the 2D pointwise model can consume directly.
    """

    def __init__(
        self,
        data_cache,
        label_cache,
        ids: Sequence[int],
        samples_per_epoch: int,
        seed: int,
        num_classes: int,
        ignore_index: Optional[int],
        balanced_sampling: bool = False,
    ):
        self.label_cache = label_cache
        self.ids = list(ids)
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.num_classes = int(num_classes)
        self.ignore_index = ignore_index
        self.epoch = 0

        if not self.ids:
            raise ValueError("RandomPixel1x1Dataset requires at least one image ID")

        self.chw_cache = build_chw_views(data_cache, label_cache, self.ids)
        self.in_ch = int(next(iter(self.chw_cache.values())).shape[0])
        self.class_index = ClassPixelIndex(label_cache, self.ids, num_classes, ignore_index) if balanced_sampling else None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return max(1, self.samples_per_epoch)

    def _sample_random_valid_pixel(self, rng: np.random.Generator) -> Tuple[int, int, int]:
        if self.class_index is not None:
            return self.class_index.sample_one(rng)

        for _ in range(1000):
            image_id = self.ids[int(rng.integers(0, len(self.ids)))]
            y = self.label_cache[image_id]
            h, w = y.shape
            yy = int(rng.integers(0, h))
            xx = int(rng.integers(0, w))
            if valid_label(int(y[yy, xx]), self.num_classes, self.ignore_index):
                return image_id, yy, xx
        raise RuntimeError("Could not sample a valid pixel after 1000 attempts")

    def __getitem__(self, index: int):
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + int(index))
        image_id, yy, xx = self._sample_random_valid_pixel(rng)
        x_chw = self.chw_cache[image_id]
        y = self.label_cache[image_id]

        x = np.ascontiguousarray(x_chw[:, yy, xx].reshape(self.in_ch, 1, 1)).astype(np.float32, copy=False)
        target = np.asarray([[int(y[yy, xx])]], dtype=np.int64)
        return torch.from_numpy(x), torch.from_numpy(target)


class RandomPixelTileDataset(Dataset):

    def __init__(
        self,
        data_cache,
        label_cache,
        ids: Sequence[int],
        patch_size: int,
        samples_per_epoch: int,
        seed: int,
        num_classes: int,
        ignore_index: Optional[int],
        balanced_sampling: bool = False,
    ):
        self.label_cache = label_cache
        self.ids = list(ids)
        self.patch_size = int(patch_size)
        self.pixels_per_tile = self.patch_size * self.patch_size
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.num_classes = int(num_classes)
        self.ignore_index = ignore_index
        self.epoch = 0

        if not self.ids:
            raise ValueError("RandomPixelTileDataset requires at least one image ID")

        self.chw_cache = build_chw_views(data_cache, label_cache, self.ids)
        self.in_ch = int(next(iter(self.chw_cache.values())).shape[0])
        self.class_index = ClassPixelIndex(label_cache, self.ids, num_classes, ignore_index) if balanced_sampling else None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return max(1, math.ceil(self.samples_per_epoch / self.pixels_per_tile))

    def _sample_random_valid_pixel(self, rng: np.random.Generator) -> Tuple[int, int, int]:
        if self.class_index is not None:
            return self.class_index.sample_one(rng)

        for _ in range(1000):
            image_id = self.ids[int(rng.integers(0, len(self.ids)))]
            y = self.label_cache[image_id]
            h, w = y.shape
            yy = int(rng.integers(0, h))
            xx = int(rng.integers(0, w))
            if valid_label(int(y[yy, xx]), self.num_classes, self.ignore_index):
                return image_id, yy, xx
        raise RuntimeError("Could not sample a valid pixel after 1000 attempts")

    def __getitem__(self, index: int):
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + int(index))
        x_tile = np.empty((self.in_ch, self.patch_size, self.patch_size), dtype=np.float32)
        y_tile = np.empty((self.patch_size, self.patch_size), dtype=np.int64)

        k = 0
        for row in range(self.patch_size):
            for col in range(self.patch_size):
                image_id, yy, xx = self._sample_random_valid_pixel(rng)
                x_chw = self.chw_cache[image_id]
                y = self.label_cache[image_id]
                x_tile[:, row, col] = x_chw[:, yy, xx]
                y_tile[row, col] = int(y[yy, xx])
                k += 1

        return torch.from_numpy(x_tile), torch.from_numpy(y_tile)


def make_train_loader(args, data_cache, label_cache, train_ids):
    if args.train_feed == "pixel1x1":
        pixel_batch_size = args.pixel_batch_size
        if pixel_batch_size is None or pixel_batch_size <= 0:
            pixel_batch_size = args.batch_size * args.patch_size * args.patch_size

        dataset = RandomPixel1x1Dataset(
            data_cache=data_cache,
            label_cache=label_cache,
            ids=train_ids,
            samples_per_epoch=args.samples_per_epoch_train,
            seed=args.seed,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
            balanced_sampling=args.balanced_sampling,
        )
        loader = DataLoader(
            dataset,
            batch_size=pixel_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        return dataset, loader

    if args.train_feed == "fake2d-tile":
        dataset = RandomPixelTileDataset(
            data_cache=data_cache,
            label_cache=label_cache,
            ids=train_ids,
            patch_size=args.patch_size,
            samples_per_epoch=args.samples_per_epoch_train,
            seed=args.seed,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
            balanced_sampling=args.balanced_sampling,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        return dataset, loader

    raise ValueError(f"Unsupported --train-feed: {args.train_feed}")


def update_confusion(confusion, pred, target, num_classes: int, ignore_index: Optional[int]):
    pred = pred.detach().view(-1).to(torch.int64).cpu()
    target = target.detach().view(-1).to(torch.int64).cpu()
    mask = (target >= 0) & (target < num_classes)
    if ignore_index is not None:
        mask &= target != ignore_index
    pred, target = pred[mask], target[mask]
    if target.numel() == 0:
        return
    idx = target * num_classes + pred.clamp(0, num_classes - 1)
    confusion += torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def confusion_to_metrics(confusion, loss_sum: float = 0.0, n_loss: int = 0) -> Metrics:
    confusion = confusion.to(torch.float64)
    correct = torch.diag(confusion).sum().item()
    total = confusion.sum().item()
    intersection = torch.diag(confusion)
    union = confusion.sum(dim=0) + confusion.sum(dim=1) - intersection
    iou = intersection / torch.clamp(union, min=1.0)
    valid = union > 0
    return Metrics(
        acc=float(correct / total) if total else 0.0,
        miou=float(iou[valid].mean().item()) if bool(valid.any()) else 0.0,
        iou=[float(v) for v in iou.tolist()],
        loss=float(loss_sum / n_loss) if n_loss else None,
    )


def set_batchnorm_eval(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def parse_checkpoint_map(entries: Optional[Sequence[str]]) -> Dict[str, Path]:
    out = {}
    for entry in entries or []:
        if "=" not in entry:
            raise ValueError("Use --fp32-checkpoints model_name=/path/to/checkpoint.pt")
        name, path = entry.split("=", 1)
        out[name.strip()] = Path(path.strip())
    return out


def checkpoint_score(path: Path):
    name = path.name.lower()
    score = 0
    score += 100 if "best" in name else 0
    score += 30 if "miou" in name else 0
    score += 20 if "acc" in name else 0
    score += 10 if "final" in name else 0
    score -= 50 if "optim" in name or "optimizer" in name else 0
    return score, path.stat().st_mtime, str(path)


def model_checkpoint_names(model_name: str) -> List[str]:
    names = [model_name]
    for alias in MODEL_CHECKPOINT_ALIASES.get(model_name, []):
        if alias not in names:
            names.append(alias)
    return names


def find_checkpoint(args, model_name: str) -> Path:
    explicit_map = parse_checkpoint_map(args.fp32_checkpoints)
    for name in model_checkpoint_names(model_name):
        if name in explicit_map:
            path = explicit_map[name]
            if not path.exists():
                raise FileNotFoundError(path)
            return path

    if args.fp32_checkpoint:
        path = Path(args.fp32_checkpoint)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    searched_dirs = []
    candidates = []
    for name in model_checkpoint_names(model_name):
        model_dir = Path(args.fp32_root) / name
        searched_dirs.append(model_dir)
        for pattern in ("*.pt", "*.pth", "*.ckpt", "*.tar"):
            candidates.extend(model_dir.glob(pattern))

    if not candidates:
        searched = ", ".join(str(p) for p in searched_dirs)
        raise FileNotFoundError(
            f"No checkpoint found under any of: {searched}. Use --fp32-checkpoint or --fp32-checkpoints."
        )
    return sorted(candidates, key=checkpoint_score, reverse=True)[0]


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint is not a mapping/state_dict")
    for key in ("state_dict", "model_state_dict", "model", "net", "network", "module"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value
    return checkpoint


def strip_prefixes(state_dict):
    out = dict(state_dict.items())
    prefixes = ("module.", "model.", "net.", "network.", "_orig_mod.")
    changed = True
    while changed:
        changed = False
        keys = list(out.keys())
        for prefix in prefixes:
            if keys and all(k.startswith(prefix) for k in keys):
                out = {k[len(prefix) :]: v for k, v in out.items()}
                changed = True
                break
    return out


def load_fp32_weights(model: nn.Module, checkpoint_path: Path, strict: bool = False) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = strip_prefixes(extract_state_dict(checkpoint))
    if strict:
        print(model.load_state_dict(state_dict, strict=True))
        return

    model_state = model.state_dict()
    matched = {
        k: v for k, v in state_dict.items()
        if k in model_state and tuple(v.shape) == tuple(model_state[k].shape)
    }
    if not matched:
        raise RuntimeError(f"No checkpoint tensors from {checkpoint_path} matched this model")
    result = model.load_state_dict(matched, strict=False)
    print(f"[weights] Loaded {len(matched)} tensors from {checkpoint_path}")
    if result.missing_keys:
        print(f"[weights] Missing keys after non-strict load: {len(result.missing_keys)}")
    if result.unexpected_keys:
        print(f"[weights] Unexpected keys after non-strict load: {len(result.unexpected_keys)}")


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, metrics: Metrics, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": asdict(metrics),
        "args": vars(args),
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, path)


def load_trainable_checkpoint(model, checkpoint_path: Path, device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(strip_prefixes(extract_state_dict(checkpoint)), strict=False)


def make_qat_processor(model, dummy_input, bitwidth: int):
    try:
        from pytorch_nndct import QatProcessor
    except Exception as exc:
        raise RuntimeError("pytorch_nndct is not available. Run inside the Vitis AI PyTorch environment.") from exc
    return QatProcessor(model, dummy_input, bitwidth=bitwidth)


def distillation_loss(student_logits, teacher_logits, temperature: float) -> torch.Tensor:
    t = float(temperature)
    return F.kl_div(
        F.log_softmax(student_logits / t, dim=1),
        F.softmax(teacher_logits / t, dim=1),
        reduction="batchmean",
    ) * (t * t)


def train_one_epoch(args, model, teacher, loader, criterion, optimizer, scaler, device, epoch: int) -> Metrics:
    model.train()
    if args.freeze_bn_after >= 0 and epoch >= args.freeze_bn_after:
        set_batchnorm_eval(model)

    if teacher is not None:
        teacher.eval()

    confusion = torch.zeros((args.num_classes, args.num_classes), dtype=torch.int64)
    loss_sum, n_loss = 0.0, 0

    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=args.use_amp and device.type == "cuda"):
            logits = model(x)
            ce_loss = criterion(logits, y)
            loss = ce_loss
            if teacher is not None and args.distill_weight > 0:
                with torch.no_grad():
                    teacher_logits = teacher(x)
                # Flatten H/W positions into samples for KL over classes.
                s = logits.permute(0, 2, 3, 1).reshape(-1, args.num_classes)
                t = teacher_logits.permute(0, 2, 3, 1).reshape(-1, args.num_classes)
                kd_loss = distillation_loss(s, t, args.distill_temperature)
                loss = ce_loss + args.distill_weight * kd_loss

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if args.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()

        pixels = int(y.numel())
        loss_sum += float(ce_loss.item()) * pixels
        n_loss += pixels
        update_confusion(confusion, torch.argmax(logits.detach(), dim=1), y, args.num_classes, args.ignore_index)

        if args.log_every > 0 and step % args.log_every == 0:
            m = confusion_to_metrics(confusion, loss_sum, n_loss)
            print(f"    step {step:04d}/{len(loader):04d} ce={m.loss:.5f} acc={m.acc:.4f} miou={m.miou:.4f}")

    return confusion_to_metrics(confusion, loss_sum, n_loss)


def pack_flat_pixels_to_tiles(x_flat: np.ndarray, y_flat: np.ndarray, start: int, batch_tiles: int, patch_size: int):
    pixels_per_tile = patch_size * patch_size
    capacity = batch_tiles * pixels_per_tile
    end = min(start + capacity, x_flat.shape[0])
    n_valid = end - start

    x_chunk = x_flat[start:end]
    y_chunk = y_flat[start:end]

    if n_valid < capacity:
        if n_valid == 0:
            raise ValueError("Cannot pack an empty pixel chunk")
        pad_n = capacity - n_valid
        x_pad = np.repeat(x_chunk[-1:], pad_n, axis=0)
        y_pad = np.repeat(y_chunk[-1:], pad_n, axis=0)
        x_chunk = np.concatenate([x_chunk, x_pad], axis=0)
        y_chunk = np.concatenate([y_chunk, y_pad], axis=0)

    # [B*P*P, C] -> [B, C, P, P]
    in_ch = x_chunk.shape[1]
    x_tiles = x_chunk.reshape(batch_tiles, patch_size, patch_size, in_ch).transpose(0, 3, 1, 2)
    y_tiles = y_chunk.reshape(batch_tiles, patch_size, patch_size)
    return np.ascontiguousarray(x_tiles), np.ascontiguousarray(y_tiles), n_valid


def evaluate_packed_pixels(model, data_cache, label_cache, ids, criterion, device, args) -> Metrics:
    model.eval()
    confusion = torch.zeros((args.num_classes, args.num_classes), dtype=torch.int64)
    loss_sum, n_loss = 0.0, 0

    with torch.no_grad():
        for image_id in ids:
            y_np = squeeze_label(label_cache[image_id])
            x_np = as_chw(data_cache[image_id], y_np)
            c, h, w = x_np.shape
            x_flat = np.ascontiguousarray(x_np.reshape(c, -1).T)  # [H*W, C]
            y_flat = np.ascontiguousarray(y_np.reshape(-1))       # [H*W]

            valid = (y_flat >= 0) & (y_flat < args.num_classes)
            if args.ignore_index is not None:
                valid &= y_flat != args.ignore_index
            x_flat = x_flat[valid]
            y_flat = y_flat[valid]
            if x_flat.shape[0] == 0:
                continue

            start = 0
            while start < x_flat.shape[0]:
                xb_np, yb_np, n_valid = pack_flat_pixels_to_tiles(
                    x_flat=x_flat,
                    y_flat=y_flat,
                    start=start,
                    batch_tiles=args.eval_batch_size,
                    patch_size=args.patch_size,
                )
                xb = torch.from_numpy(xb_np).float().to(device, non_blocking=True)
                yb = torch.from_numpy(yb_np).long().to(device, non_blocking=True)
                logits = model(xb)

                logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, args.num_classes)[:n_valid]
                y_flat_t = yb.reshape(-1)[:n_valid]
                loss = criterion(logits_flat, y_flat_t)
                pred = torch.argmax(logits_flat, dim=1)

                loss_sum += float(loss.item()) * n_valid
                n_loss += int(n_valid)
                update_confusion(confusion, pred, y_flat_t, args.num_classes, args.ignore_index)
                start += n_valid

    return confusion_to_metrics(confusion, loss_sum, n_loss)


def _valid_target_mask_np(y: np.ndarray, num_classes: int, ignore_index: Optional[int]) -> np.ndarray:
    mask = (y >= 0) & (y < num_classes)
    if ignore_index is not None:
        mask &= y != ignore_index
    return mask


def _sanitize_targets_for_ce(y: np.ndarray, num_classes: int, ignore_index: int) -> Tuple[np.ndarray, np.ndarray]:
    valid = _valid_target_mask_np(y, num_classes, ignore_index)
    y_out = y.astype(np.int64, copy=True)
    y_out[~valid] = ignore_index
    return y_out, valid


def evaluate_spatial_patches(model, data_cache, label_cache, ids, criterion, device, args) -> Metrics:
    model.eval()
    confusion = torch.zeros((args.num_classes, args.num_classes), dtype=torch.int64)
    loss_sum, n_loss = 0.0, 0
    p = int(args.patch_size)
    fill_ignore = int(args.ignore_index if args.ignore_index is not None else -100)

    with torch.no_grad():
        for image_id in ids:
            y_np_raw = squeeze_label(label_cache[image_id])
            x_np = as_chw(data_cache[image_id], y_np_raw)
            c, h, w = x_np.shape

            coords = [(top, left) for top in range(0, h, p) for left in range(0, w, p)]
            for start in range(0, len(coords), args.eval_batch_size):
                batch_coords = coords[start : start + args.eval_batch_size]
                b = len(batch_coords)
                xb_np = np.empty((b, c, p, p), dtype=np.float32)
                yb_np = np.full((b, p, p), fill_ignore, dtype=np.int64)
                valid_count = 0

                for bi, (top, left) in enumerate(batch_coords):
                    hh = min(p, h - top)
                    ww = min(p, w - left)
                    xb_np[bi].fill(0.0)
                    xb_np[bi, :, :hh, :ww] = x_np[:, top : top + hh, left : left + ww]

                    y_patch = y_np_raw[top : top + hh, left : left + ww]
                    y_clean, valid = _sanitize_targets_for_ce(y_patch, args.num_classes, fill_ignore)
                    yb_np[bi, :hh, :ww] = y_clean
                    valid_count += int(valid.sum())

                if valid_count == 0:
                    continue

                xb = torch.from_numpy(np.ascontiguousarray(xb_np)).float().to(device, non_blocking=True)
                yb = torch.from_numpy(yb_np).long().to(device, non_blocking=True)
                logits = model(xb)
                loss = criterion(logits, yb)

                loss_sum += float(loss.item()) * valid_count
                n_loss += int(valid_count)
                update_confusion(confusion, torch.argmax(logits, dim=1), yb, args.num_classes, fill_ignore)

    return confusion_to_metrics(confusion, loss_sum, n_loss)


def evaluate_model(model, data_cache, label_cache, ids, criterion, device, args) -> Metrics:
    if args.eval_mode == "patches":
        return evaluate_spatial_patches(model, data_cache, label_cache, ids, criterion, device, args)
    if args.eval_mode == "packed_pixels":
        return evaluate_packed_pixels(model, data_cache, label_cache, ids, criterion, device, args)
    raise ValueError(f"Unsupported --eval-mode: {args.eval_mode}")


def export_input_from_data(args, data_cache, label_cache, ids) -> torch.Tensor:
    image_id = ids[0] if ids else next(iter(data_cache.keys()))
    y = squeeze_label(label_cache[image_id])
    x = as_chw(data_cache[image_id], y)
    _, h, w = x.shape
    p = args.patch_size

    if h < p or w < p:
        pad_h = max(0, p - h)
        pad_w = max(0, p - w)
        x = np.pad(x, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
        h, w = x.shape[1:]

    top = 0 if h == p else min((h - p) // 2, h - p)
    left = 0 if w == p else min((w - p) // 2, w - p)
    patch = np.ascontiguousarray(x[:, top : top + p, left : left + p])
    return torch.from_numpy(patch).float().unsqueeze(0)


def export_xmodel(args, qat_processor, quantized_model, output_dir: Path, export_input: torch.Tensor) -> List[str]:
    deployable_dir = output_dir / "deployable"
    deployable_dir.mkdir(parents=True, exist_ok=True)

    quantized_model.eval().cpu()
    print(f"[deploy] Converting trainable QAT model to deployable model: {deployable_dir}")
    deployable_model = qat_processor.to_deployable(quantized_model, str(deployable_dir))
    deployable_model.eval().cpu()

    print(f"[deploy] CPU forward for export input shape {tuple(export_input.shape)}")
    with torch.no_grad():
        out = deployable_model(export_input.cpu())
    print(f"[deploy] Deployable output shape: {tuple(out.shape)}")

    print("[deploy] Exporting xmodel")
    try:
        qat_processor.export_xmodel(str(deployable_dir), deploy_check=args.deploy_check)
    except TypeError:
        qat_processor.export_xmodel(str(deployable_dir))

    xmodels = sorted(str(p) for p in deployable_dir.glob("**/*.xmodel"))
    for path in xmodels:
        print(f"[deploy] xmodel: {path}")
    if not xmodels:
        print(f"[deploy] Warning: no .xmodel found under {deployable_dir}")
    return xmodels


def run_qat(args, data_cache, label_cache, train_ids, val_ids, test_ids, in_ch, device) -> RunSummary:
    print("\n" + "=" * 80)
    print(f"[model] {args.model}")
    print("=" * 80)

    fp32_ckpt = find_checkpoint(args, args.model)
    print(f"[weights] FP32 checkpoint: {fp32_ckpt}")

    dataset_short = f"{Path(args.dataset_name).name}_P{args.patch_size}"
    out_dir = Path(args.out_root) / dataset_short / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    teacher_model = build_model(model_name=args.model, in_ch=in_ch, num_classes=args.num_classes)
    load_fp32_weights(teacher_model, fp32_ckpt, strict=args.strict_fp32_load)
    teacher_model.to(device).eval()

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_index)
    fp32_val = evaluate_model(teacher_model, data_cache, label_cache, val_ids, criterion, device, args)
    print(f"[fp32 val] loss={fp32_val.loss:.5f} acc={fp32_val.acc:.4f} miou={fp32_val.miou:.4f}")

    float_model = build_model(model_name=args.model, in_ch=in_ch, num_classes=args.num_classes)
    load_fp32_weights(float_model, fp32_ckpt, strict=args.strict_fp32_load)
    float_model.train()

    dummy_input = torch.randn(1, in_ch, args.patch_size, args.patch_size)
    print(f"[qat] QatProcessor dummy/export shape: {tuple(dummy_input.shape)}")
    qat_processor = make_qat_processor(float_model, dummy_input, args.bitwidth)
    quantized_model = qat_processor.trainable_model().to(device)

    optimizer = torch.optim.AdamW(quantized_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.cosine_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs),
            eta_min=args.min_lr,
        )
    scaler = GradScaler(enabled=args.use_amp and device.type == "cuda")
    train_dataset, train_loader = make_train_loader(args, data_cache, label_cache, train_ids)

    if args.train_feed == "pixel1x1":
        effective_pixels = int(train_loader.batch_size or 0)
        feed_desc = f"pixel1x1, pixel_batch_size={effective_pixels}"
    else:
        effective_pixels = args.batch_size * args.patch_size * args.patch_size
        feed_desc = f"fake2d-tile, tiles/step={args.batch_size}"
    print(
        f"[qat] epochs={args.epochs}, train_feed={feed_desc}, "
        f"effective_pixels/step={effective_pixels}, samples/epoch={args.samples_per_epoch_train}"
    )
    print(f"[eval] eval_mode={args.eval_mode}, eval_batch_size={args.eval_batch_size}, patch_size={args.patch_size}")
    print(
        f"[qat] lr={args.lr}, weight_decay={args.weight_decay}, freeze_bn_after={args.freeze_bn_after}, "
        f"distill_weight={args.distill_weight}"
    )

    best_metric, best_epoch = -1.0, -1
    best_val = Metrics(0.0, 0.0, [0.0] * args.num_classes, None)
    final_val = Metrics(0.0, 0.0, [0.0] * args.num_classes, None)
    best_path = out_dir / "qat_trainable_best.pth"
    latest_path = out_dir / "qat_trainable_latest.pth"

    teacher_for_train = teacher_model if args.distill_weight > 0 else None

    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        t0 = time.time()
        train_metrics = train_one_epoch(
            args=args,
            model=quantized_model,
            teacher=teacher_for_train,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
        )
        if scheduler is not None:
            scheduler.step()
        final_val = evaluate_model(quantized_model, data_cache, label_cache, val_ids, criterion, device, args)

        score = final_val.acc if args.select_best_by == "acc" else final_val.miou
        print(
            f"[epoch {epoch:03d}] "
            f"train ce={train_metrics.loss:.5f} acc={train_metrics.acc:.4f} miou={train_metrics.miou:.4f} | "
            f"val ce={final_val.loss:.5f} acc={final_val.acc:.4f} miou={final_val.miou:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | {time.time() - t0:.1f}s"
        )

        save_checkpoint(latest_path, quantized_model, optimizer, scheduler, epoch, final_val, args)
        if score > best_metric:
            best_metric, best_epoch, best_val = score, epoch, final_val
            shutil.copy2(latest_path, best_path)
            print(f"[checkpoint] New best by {args.select_best_by}: {best_path}")

    if args.epochs == 0:
        final_val = evaluate_model(quantized_model, data_cache, label_cache, val_ids, criterion, device, args)
        best_val, best_epoch = final_val, 0
        save_checkpoint(latest_path, quantized_model, optimizer, scheduler, 0, final_val, args)
        shutil.copy2(latest_path, best_path)

    if args.use_best_for_export and best_path.exists():
        print(f"[checkpoint] Loading best QAT checkpoint for export/eval: {best_path}")
        load_trainable_checkpoint(quantized_model, best_path, device)
        final_val = evaluate_model(quantized_model, data_cache, label_cache, val_ids, criterion, device, args)

    final_test = None
    if args.evaluate_test:
        final_test = evaluate_model(quantized_model, data_cache, label_cache, test_ids, criterion, device, args)
        print(f"[test] loss={final_test.loss:.5f} acc={final_test.acc:.4f} miou={final_test.miou:.4f}")

    xmodels, deployable_dir = [], None
    if args.export_xmodel:
        xmodels = export_xmodel(
            args=args,
            qat_processor=qat_processor,
            quantized_model=quantized_model,
            output_dir=out_dir,
            export_input=export_input_from_data(args, data_cache, label_cache, val_ids or test_ids or train_ids),
        )
        deployable_dir = str(out_dir / "deployable")

    summary = RunSummary(
        model_name=args.model,
        fp32_checkpoint=str(fp32_ckpt),
        output_dir=str(out_dir),
        best_qat_checkpoint=str(best_path),
        latest_qat_checkpoint=str(latest_path),
        deployable_dir=deployable_dir,
        xmodel_files=xmodels,
        fp32_val=fp32_val,
        best_val=best_val,
        best_epoch=int(best_epoch),
        final_val=final_val,
        final_test=final_test,
        effective_pixels_per_step=effective_pixels,
    )
    with open(out_dir / "qat_pointwise_1d_export2d_summary.json", "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Vitis AI QAT for pointwise hyperspectral model using literal 1x1 pixel QAT, patch validation/test, and 2D export.")
    p.add_argument("--model", default="spectralpointwise2d", choices=POINTWISE_MODELS)
    p.add_argument("--dataset-name", default="dataset_name")
    p.add_argument("--data-root", default="datasets")
    p.add_argument("--n-images", type=int, default=60)
    p.add_argument("--val-ids", default=",".join(str(i) for i in DEFAULT_VAL_IDS))
    p.add_argument("--test-ids", default=",".join(str(i) for i in DEFAULT_TEST_IDS))
    p.add_argument("--patch-size", type=int, default=32, help="2D export input H/W and random-pixel tile H/W.")
    p.add_argument("--num-classes", type=int, default=3)
    p.add_argument("--ignore-index", type=int, default=-100)
    p.add_argument("--mmap-mode", default="r", choices=["r", "none"], help="Default r keeps .npy arrays memory-mapped. Use none only if you have enough RAM to preload the full dataset.")

    p.add_argument("--fp32-root", default="weights_P32")
    p.add_argument("--fp32-checkpoint", default=None)
    p.add_argument("--fp32-checkpoints", action="append", default=None)
    p.add_argument("--strict-fp32-load", action="store_true")

    p.add_argument("--out-root", default="out_folder")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--train-feed", default="pixel1x1", choices=["pixel1x1", "fake2d-tile"],
                   help="pixel1x1 feeds [B,C,1,1] random pixels; fake2d-tile packs independent pixels into [B,C,P,P].")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Tiles per optimizer step for --train-feed fake2d-tile. Effective pixels = batch_size * patch_size^2.")
    p.add_argument("--pixel-batch-size", type=int, default=4096,
                   help="Pixels per optimizer step for --train-feed pixel1x1. Use 0 to derive from batch_size * patch_size^2.")
    p.add_argument("--eval-mode", default="patches", choices=["patches", "packed_pixels"],
                   help="patches scans full images as spatial patches; packed_pixels flattens and repacks valid pixels.")
    p.add_argument("--eval-batch-size", type=int, default=32, help="Patches/tiles per eval forward pass.")
    p.add_argument("--samples-per-epoch-train", type=int, default=300000, help="Number of random pixels sampled per epoch.")
    p.add_argument("--balanced-sampling", action="store_true", help="Sample classes uniformly. This stores per-class pixel coordinates and uses more RAM; leave off for lowest memory.")

    p.add_argument("--min-lr", type=float, default=5e-6)
    p.add_argument("--cosine-lr", action="store_true", default=True)
    p.add_argument("--no-cosine-lr", dest="cosine_lr", action="store_false")
    p.add_argument("--freeze-bn-after", type=int, default=-1, help="Set BN layers to eval from this epoch onward. Use -1 to disable.")
    p.add_argument("--distill-weight", type=float, default=0.10, help="KL distillation from the FP32 model. Set 0 to disable.")
    p.add_argument("--distill-temperature", type=float, default=2.0)
    p.add_argument("--grad-clip-norm", type=float, default=0.0)
    p.add_argument("--use-amp", action="store_true", help="Usually leave off for QAT unless you have tested it in your Vitis AI stack.")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--select-best-by", default="acc", choices=["acc", "miou"])

    p.add_argument("--bitwidth", type=int, default=8)
    p.add_argument("--export-xmodel", action="store_true")
    p.add_argument("--deploy-check", action="store_true")
    p.add_argument("--use-best-for-export", action="store_true", default=True)
    p.add_argument("--use-latest-for-export", dest="use_best_for_export", action="store_false")
    p.add_argument("--evaluate-test", action="store_true", default=True)
    p.add_argument("--no-evaluate-test", dest="evaluate_test", action="store_false")



    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)



    return p


def main() -> None:
    args = build_argparser().parse_args()
    seed_everything(args.seed, deterministic=args.deterministic)

    val_ids, test_ids = parse_id_list(args.val_ids), parse_id_list(args.test_ids)
    all_ids, train_ids, val_ids, test_ids = build_fixed_split(args.n_images, val_ids, test_ids)
    base = Path(args.data_root) / args.dataset_name

    print(f"[split] Train={len(train_ids)} Val={len(val_ids)} Test={len(test_ids)}")
    print(f"[split] Train IDs: {train_ids}")
    print(f"[split] Val IDs  : {val_ids}")
    print(f"[split] Test IDs : {test_ids}")
    print(f"[data] Dataset: {base}")

    mmap_mode = None if args.mmap_mode == "none" else args.mmap_mode
    if mmap_mode is None:
        print("[cache] Warning: --mmap-mode none preloads all hyperspectral cubes into RAM. Use --mmap-mode r for low-memory execution.")
    data_cache, label_cache = load_data_cache(base, all_ids, mmap_mode=mmap_mode)
    chw_cache = build_chw_views(data_cache, label_cache, [train_ids[0]])
    in_ch = infer_in_channels(chw_cache, train_ids[0])
    print(f"[data] Inferred input channels: {in_ch}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    summary = run_qat(args, data_cache, label_cache, train_ids, val_ids, test_ids, in_ch, device)

    print("\n[done] Pointwise 1D-style QAT completed")
    print(f"[done] best_epoch={summary.best_epoch}")
    print(f"[done] best_val acc={summary.best_val.acc:.4f} miou={summary.best_val.miou:.4f}")
    if summary.final_test is not None:
        print(f"[done] test acc={summary.final_test.acc:.4f} miou={summary.final_test.miou:.4f}")
    print(f"[done] summary: {Path(summary.output_dir) / 'qat_pointwise_1d_export2d_summary.json'}")


if __name__ == "__main__":
    main()
