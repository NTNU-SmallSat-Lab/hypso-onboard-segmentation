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
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

try:
    import pytorch_nndct  # noqa: F401
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from models_vitis_ai_qat import build_model


PATCH_2D_MODELS = [
    "spectralpointwise2d",
    "spw_unet_large",
    "spw_unet_medium",
    "spw_unet_small",
    "justounetsimple",
]

MODEL_CHECKPOINT_ALIASES = {
    "spectralpointwise2d": [
        "spectralpointwise2d",
        "SpectralPointwise2D",
        "spectralpointwise1D",
        "SpectralPointwise1D",
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
class ModelRunSummary:
    model_name: str
    fp32_checkpoint: str
    output_dir: str
    best_qat_checkpoint: str
    latest_qat_checkpoint: str
    deployable_dir: Optional[str]
    xmodel_files: List[str]
    best_val_miou: float
    best_epoch: int
    final_val: Metrics
    final_test: Optional[Metrics]


def parse_id_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def load_data_cache(base: Path, ids: Sequence[int], mmap_mode: str = "r"):
    data_cache, label_cache = {}, {}
    print(f"[cache] Loading {len(ids)} files from {base} (mmap_mode={mmap_mode})")
    for i in ids:
        data_path = base / f"data{i}.npy"
        label_path = base / f"label{i}.npy"
        if not data_path.exists():
            raise FileNotFoundError(data_path)
        if not label_path.exists():
            raise FileNotFoundError(label_path)

        data = np.load(data_path, mmap_mode=mmap_mode)
        label = squeeze_label(np.load(label_path, mmap_mode=mmap_mode))
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        if label.dtype != np.int64:
            label = label.astype(np.int64)
        data_cache[i], label_cache[i] = data, label
    return data_cache, label_cache


def infer_in_channels(data_cache, label_cache, image_id: int) -> int:
    return int(as_chw(data_cache[image_id], label_cache[image_id]).shape[0])


def pad_if_needed(x: np.ndarray, y: np.ndarray, patch_size: int):
    _, h, w = x.shape
    pad_h, pad_w = max(0, patch_size - h), max(0, patch_size - w)
    if pad_h == 0 and pad_w == 0:
        return x, y
    x = np.pad(x, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
    y = np.pad(y, ((0, pad_h), (0, pad_w)), mode="edge")
    return x, y


def crop_patch(x: np.ndarray, y: np.ndarray, patch_size: int, rng: np.random.Generator):
    x, y = pad_if_needed(x, y, patch_size)
    _, h, w = x.shape
    top = 0 if h == patch_size else int(rng.integers(0, h - patch_size + 1))
    left = 0 if w == patch_size else int(rng.integers(0, w - patch_size + 1))
    return (
        np.ascontiguousarray(x[:, top : top + patch_size, left : left + patch_size]),
        np.ascontiguousarray(y[top : top + patch_size, left : left + patch_size]),
    )


class RandomPatchDataset(Dataset):
    def __init__(self, data_cache, label_cache, ids, patch_size, samples_per_epoch, seed):
        self.data_cache = data_cache
        self.label_cache = label_cache
        self.ids = list(ids)
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.epoch = 0
        if not self.ids:
            raise ValueError("RandomPatchDataset requires at least one image ID")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int):
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index)
        image_id = self.ids[int(rng.integers(0, len(self.ids)))]
        y = squeeze_label(self.label_cache[image_id])
        x = as_chw(self.data_cache[image_id], y)
        xp, yp = crop_patch(x, y, self.patch_size, rng)
        return torch.from_numpy(xp).float(), torch.from_numpy(yp).long()


def make_train_loader(args, data_cache, label_cache, train_ids):
    dataset = RandomPatchDataset(
        data_cache=data_cache,
        label_cache=label_cache,
        ids=train_ids,
        patch_size=args.patch_size,
        samples_per_epoch=args.samples_per_epoch_train,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    return dataset, loader


def update_confusion(confusion, pred, target, num_classes: int, ignore_index: int):
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
    correct, total = torch.diag(confusion).sum().item(), confusion.sum().item()
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


def stitch_starts(length: int, patch_size: int, stride: int) -> List[int]:
    if length <= patch_size:
        return [0]
    last = length - patch_size
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def stitch_patch(x: np.ndarray, top: int, left: int, patch_size: int):
    _, h, w = x.shape
    valid_h, valid_w = min(patch_size, h - top), min(patch_size, w - left)
    patch = x[:, top : top + valid_h, left : left + valid_w]
    pad_h, pad_w = patch_size - valid_h, patch_size - valid_w
    if pad_h > 0 or pad_w > 0:
        patch = np.pad(patch, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
    return np.ascontiguousarray(patch), valid_h, valid_w


def write_patch_logits(stitched_logits, filled, patch_logits, top: int, left: int, valid_h: int, valid_w: int):
    patch_logits = patch_logits[:, :valid_h, :valid_w]
    region_filled = filled[top : top + valid_h, left : left + valid_w]
    write_mask = ~region_filled
    if bool(write_mask.any()):
        target = stitched_logits[:, top : top + valid_h, left : left + valid_w]
        target[:, write_mask] = patch_logits[:, write_mask]
        region_filled[write_mask] = True


def run_patch_batch(model, patch_buffer, meta_buffer, stitched_logits, filled, device):
    if not patch_buffer:
        return
    xb = torch.from_numpy(np.stack(patch_buffer, axis=0)).float().to(device, non_blocking=True)
    logits_b = model(xb).detach().cpu()
    if logits_b.ndim != 4:
        raise RuntimeError(f"Expected logits [B,C,H,W], got {tuple(logits_b.shape)}")
    for j, (top, left, valid_h, valid_w) in enumerate(meta_buffer):
        write_patch_logits(stitched_logits, filled, logits_b[j], top, left, valid_h, valid_w)


def evaluate_stitched(model, data_cache, label_cache, ids, criterion, device, args) -> Metrics:
    model.eval()
    stride = args.stitch_stride or args.patch_size
    confusion = torch.zeros((args.num_classes, args.num_classes), dtype=torch.int64)
    loss_sum, n_loss = 0.0, 0

    with torch.no_grad():
        for image_id in ids:
            y_np = squeeze_label(label_cache[image_id])
            x_np = as_chw(data_cache[image_id], y_np)
            _, h, w = x_np.shape

            stitched_logits = torch.zeros((args.num_classes, h, w), dtype=torch.float32)
            filled = torch.zeros((h, w), dtype=torch.bool)
            patches, metas = [], []

            for top in stitch_starts(h, args.patch_size, stride):
                for left in stitch_starts(w, args.patch_size, stride):
                    patch, valid_h, valid_w = stitch_patch(x_np, top, left, args.patch_size)
                    patches.append(patch)
                    metas.append((top, left, valid_h, valid_w))
                    if len(patches) >= args.eval_batch_size:
                        run_patch_batch(model, patches, metas, stitched_logits, filled, device)
                        patches, metas = [], []

            run_patch_batch(model, patches, metas, stitched_logits, filled, device)
            if not bool(filled.all()):
                raise RuntimeError(f"Stitched validation missed {(~filled).sum().item()} pixels in image {image_id}")

            logits = stitched_logits.unsqueeze(0).to(device, non_blocking=True)
            y = torch.from_numpy(np.ascontiguousarray(y_np)).long().unsqueeze(0).to(device, non_blocking=True)
            loss = criterion(logits, y)
            pred = torch.argmax(logits, dim=1)

            pixels = int(y.numel())
            loss_sum += float(loss.item()) * pixels
            n_loss += pixels
            update_confusion(confusion, pred, y, args.num_classes, args.ignore_index)

    return confusion_to_metrics(confusion, loss_sum, n_loss)


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
        if args.model == "all":
            raise ValueError("--fp32-checkpoint can only be used with one --model")
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
            f"No checkpoint found under any of: {searched}. "
            "Use --fp32-checkpoint or --fp32-checkpoints."
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
    matched = {k: v for k, v in state_dict.items() if k in model_state and tuple(v.shape) == tuple(model_state[k].shape)}
    if not matched:
        raise RuntimeError(f"No checkpoint tensors from {checkpoint_path} matched this model")
    result = model.load_state_dict(matched, strict=False)
    print(f"[weights] Loaded {len(matched)} tensors from {checkpoint_path}")
    if result.missing_keys:
        print(f"[weights] Missing keys after non-strict load: {len(result.missing_keys)}")
    if result.unexpected_keys:
        print(f"[weights] Unexpected keys after non-strict load: {len(result.unexpected_keys)}")


def save_checkpoint(path: Path, model, optimizer, epoch: int, metrics: Metrics, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": asdict(metrics),
            "args": vars(args),
        },
        path,
    )


def load_trainable_checkpoint(model, checkpoint_path: Path, device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(strip_prefixes(extract_state_dict(checkpoint)), strict=False)


def make_qat_processor(model, dummy_input, bitwidth: int):
    try:
        from pytorch_nndct import QatProcessor
    except Exception as exc:
        raise RuntimeError("pytorch_nndct is not available. Run inside the Vitis AI PyTorch environment.") from exc
    return QatProcessor(model, dummy_input, bitwidth=bitwidth)


def train_one_epoch(args, model, loader, criterion, optimizer, scaler, device) -> Metrics:
    model.train()
    confusion = torch.zeros((args.num_classes, args.num_classes), dtype=torch.int64)
    loss_sum, n_loss = 0.0, 0

    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=args.use_amp and device.type == "cuda"):
            logits = model(x)
            loss = criterion(logits, y)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        pixels = int(y.numel())
        loss_sum += float(loss.item()) * pixels
        n_loss += pixels
        update_confusion(confusion, torch.argmax(logits.detach(), dim=1), y, args.num_classes, args.ignore_index)

        if args.log_every > 0 and step % args.log_every == 0:
            m = confusion_to_metrics(confusion, loss_sum, n_loss)
            print(f"    step {step:04d}/{len(loader):04d} loss={m.loss:.5f} acc={m.acc:.4f} miou={m.miou:.4f}")

    return confusion_to_metrics(confusion, loss_sum, n_loss)


def export_input_from_data(args, data_cache, label_cache, ids) -> torch.Tensor:
    image_id = ids[0] if ids else next(iter(data_cache.keys()))
    y = squeeze_label(label_cache[image_id])
    x = as_chw(data_cache[image_id], y)
    x_patch, _ = crop_patch(x, y, args.patch_size, np.random.default_rng(args.seed + 999_999))
    return torch.from_numpy(x_patch).float().unsqueeze(0)


def export_xmodel(args, qat_processor, quantized_model, output_dir: Path, export_input: torch.Tensor) -> List[str]:
    deployable_dir = output_dir / "deployable"
    deployable_dir.mkdir(parents=True, exist_ok=True)

    quantized_model.eval().cpu()
    print(f"[deploy] Converting to deployable model: {deployable_dir}")
    deployable_model = qat_processor.to_deployable(quantized_model, str(deployable_dir))
    deployable_model.eval().cpu()

    print("[deploy] Running one CPU forward pass with batch size 1")
    with torch.no_grad():
        _ = deployable_model(export_input.cpu())

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


def run_qat_for_model(args, model_name, dataset_short, data_cache, label_cache, train_ids, val_ids, test_ids, in_ch, device):
    print("\n" + "=" * 80)
    print(f"[model] {model_name}")
    print("=" * 80)

    fp32_ckpt = find_checkpoint(args, model_name)
    print(f"[weights] FP32 checkpoint: {fp32_ckpt}")

    out_dir = Path(args.out_root) / dataset_short / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    float_model = build_model(model_name=model_name, in_ch=in_ch, num_classes=args.num_classes)
    load_fp32_weights(float_model, fp32_ckpt, strict=args.strict_fp32_load)
    float_model.train()

    dummy_input = torch.randn(1, in_ch, args.patch_size, args.patch_size)
    qat_processor = make_qat_processor(float_model, dummy_input, args.bitwidth)
    quantized_model = qat_processor.trainable_model().to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_index)
    optimizer = torch.optim.AdamW(quantized_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.use_amp and device.type == "cuda")
    train_dataset, train_loader = make_train_loader(args, data_cache, label_cache, train_ids)

    best_miou, best_epoch = -1.0, -1
    best_path = out_dir / "qat_trainable_best.pth"
    latest_path = out_dir / "qat_trainable_latest.pth"
    final_val = Metrics(0.0, 0.0, [0.0] * args.num_classes, None)

    print(f"[qat] epochs={args.epochs}, batch={args.batch_size}, samples/epoch={args.samples_per_epoch_train}, validation=stitch")
    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        t0 = time.time()
        train_metrics = train_one_epoch(args, quantized_model, train_loader, criterion, optimizer, scaler, device)
        final_val = evaluate_stitched(quantized_model, data_cache, label_cache, val_ids, criterion, device, args)
        print(
            f"[epoch {epoch:03d}] "
            f"train loss={train_metrics.loss:.5f} acc={train_metrics.acc:.4f} miou={train_metrics.miou:.4f} | "
            f"val loss={final_val.loss:.5f} acc={final_val.acc:.4f} miou={final_val.miou:.4f} | "
            f"{time.time() - t0:.1f}s"
        )
        save_checkpoint(latest_path, quantized_model, optimizer, epoch, final_val, args)
        if final_val.miou > best_miou:
            best_miou, best_epoch = final_val.miou, epoch
            shutil.copy2(latest_path, best_path)
            print(f"[checkpoint] New best QAT checkpoint: {best_path}")

    if args.epochs == 0:
        final_val = evaluate_stitched(quantized_model, data_cache, label_cache, val_ids, criterion, device, args)
        best_miou, best_epoch = final_val.miou, 0
        save_checkpoint(latest_path, quantized_model, optimizer, 0, final_val, args)
        shutil.copy2(latest_path, best_path)

    if args.use_best_for_export and best_path.exists():
        print(f"[checkpoint] Loading best QAT checkpoint for export: {best_path}")
        load_trainable_checkpoint(quantized_model, best_path, device)

    final_test = None
    if args.evaluate_test:
        final_test = evaluate_stitched(quantized_model, data_cache, label_cache, test_ids, criterion, device, args)
        print(f"[test] loss={final_test.loss:.5f} acc={final_test.acc:.4f} miou={final_test.miou:.4f}")

    xmodels, deployable_dir = [], None
    if args.export_xmodel:
        xmodels = export_xmodel(
            args,
            qat_processor,
            quantized_model,
            out_dir,
            export_input_from_data(args, data_cache, label_cache, val_ids or test_ids or train_ids),
        )
        deployable_dir = str(out_dir / "deployable")

    summary = ModelRunSummary(
        model_name=model_name,
        fp32_checkpoint=str(fp32_ckpt),
        output_dir=str(out_dir),
        best_qat_checkpoint=str(best_path),
        latest_qat_checkpoint=str(latest_path),
        deployable_dir=deployable_dir,
        xmodel_files=xmodels,
        best_val_miou=float(best_miou),
        best_epoch=int(best_epoch),
        final_val=final_val,
        final_test=final_test,
    )
    with open(out_dir / "qat_summary.json", "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reduced Vitis AI QAT + .xmodel export for patch-based 2D models.")
    p.add_argument("--model", default="spw_unet_small", choices=["all"] + PATCH_2D_MODELS)
    p.add_argument("--dataset-name", default="dataset_name")
    p.add_argument("--data-root", default="datasets")
    p.add_argument("--n-images", type=int, default=60)
    p.add_argument("--val-ids", default=",".join(str(i) for i in DEFAULT_VAL_IDS))
    p.add_argument("--test-ids", default=",".join(str(i) for i in DEFAULT_TEST_IDS))
    p.add_argument("--patch-size", type=int, default=32)
    p.add_argument("--num-classes", type=int, default=3)
    p.add_argument("--ignore-index", type=int, default=-100)
    p.add_argument("--mmap-mode", default="r", choices=["r", "none"], help="Default is r. Use none to fully load .npy arrays.")

    p.add_argument("--fp32-root", default="weights_P32")
    p.add_argument("--fp32-checkpoint", default=None)
    p.add_argument("--fp32-checkpoints", action="append", default=None)
    p.add_argument("--strict-fp32-load", action="store_true")

    p.add_argument("--out-root", default="out_folder")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--stitch-stride", type=int, default=None)
    p.add_argument("--use-amp", action="store_true")
    p.add_argument("--log-every", type=int, default=25)

    p.add_argument("--bitwidth", type=int, default=8)
    p.add_argument("--export-xmodel", action="store_true", default=True)
    p.add_argument("--deploy-check", action="store_true")
    p.add_argument("--use-best-for-export", action="store_true", default=True)
    p.add_argument("--use-latest-for-export", dest="use_best_for_export", action="store_false")
    p.add_argument("--evaluate-test", action="store_true", default=True)
    p.add_argument("--no-evaluate-test", dest="evaluate_test", action="store_false")

    
#    Method 2
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--samples-per-epoch-train", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)


#    Method 1
#    p.add_argument("--epochs", type=int, default=20)
#    p.add_argument("--samples-per-epoch-train", type=int, default=2000)
#    p.add_argument("--lr", type=float, default=1e-4)
#    p.add_argument("--weight-decay", type=float, default=1e-4)


    return p


def main() -> None:
    args = build_argparser().parse_args()
    seed_everything(args.seed)

    val_ids, test_ids = parse_id_list(args.val_ids), parse_id_list(args.test_ids)
    all_ids, train_ids, val_ids, test_ids = build_fixed_split(args.n_images, val_ids, test_ids)
    dataset_short = f"{Path(args.dataset_name).name}_P{args.patch_size}"
    base = Path(args.data_root) / args.dataset_name

    print(f"[split] Train={len(train_ids)} Val={len(val_ids)} Test={len(test_ids)}")
    print(f"[split] Train IDs: {train_ids}")
    print(f"[split] Val IDs  : {val_ids}")
    print(f"[split] Test IDs : {test_ids}")
    print(f"[data] Dataset: {base}")

    mmap_mode = None if args.mmap_mode == "none" else args.mmap_mode
    data_cache, label_cache = load_data_cache(base, all_ids, mmap_mode=mmap_mode)
    in_ch = infer_in_channels(data_cache, label_cache, train_ids[0])
    print(f"[data] Inferred input channels: {in_ch}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    models = PATCH_2D_MODELS if args.model == "all" else [args.model]
    summaries = [
        run_qat_for_model(args, name, dataset_short, data_cache, label_cache, train_ids, val_ids, test_ids, in_ch, device)
        for name in models
    ]

    out_root = Path(args.out_root) / dataset_short
    out_root.mkdir(parents=True, exist_ok=True)
    summary_file = out_root / "qat_all_summaries.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in summaries], f, indent=2)

    print("\n[done] QAT runs completed")
    print(f"[done] Summary: {summary_file}")
    for s in summaries:
        print(f"  {s.model_name}: best_val_miou={s.best_val_miou:.4f}, best_epoch={s.best_epoch}, xmodels={len(s.xmodel_files)}")


if __name__ == "__main__":
    main()
