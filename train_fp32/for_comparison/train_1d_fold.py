from pathlib import Path
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import time
import copy
from torch.amp import GradScaler, autocast

from model_loader import build_model


def to_python(obj):
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python(v) for v in obj]
    return obj


def seed_everything(seed: int, deterministic: bool, allow_tf32: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)

    if allow_tf32:
        torch.set_float32_matmul_precision("high")
    else:
        torch.set_float32_matmul_precision("highest")


def make_worker_init_fn(seed: int):
    def seed_worker(worker_id: int):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return seed_worker


def train_1d_fold(
    model_name: str,
    dataset_name: str,
    base: Path,
    out: Path,
    train_ids: list,
    val_ids: list,
    test_ids: list,
    SAVE_FROM_EPOCH: int,
    BATCH: int,
    EPOCHS: int,
    NCLASSES: int,
    LR: float,
    WEIGHT_DECAY: float,
    SAMPLES_PER_EPOCH_TRAIN: int,
    SAMPLES_PER_EPOCH_VAL: int,
    SEED: int,
    data_cache: dict,
    label_cache: dict,
    VALIDATION_MODE: str = "patch",      # "patch"/"pixel"/"sample" or "full"
    USE_AMP: bool = True,                 # mixed precision training/patch validation for speed
    DETERMINISTIC: bool = False,          # more repeatable runs, usually slower
    ALLOW_TF32: bool = True,              # faster on Ampere+, less bitwise-stable
    FORCE_TRAIN_WORKERS_ZERO: bool = None # None -> True when deterministic, else False
):
    base = Path(base)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    VALIDATION_MODE = str(VALIDATION_MODE).lower()
    if VALIDATION_MODE in {"pixel", "sample", "fixed", "fixed_pixel"}:
        VALIDATION_MODE = "patch"

    if VALIDATION_MODE not in {"patch", "full"}:
        raise ValueError(
            "VALIDATION_MODE must be 'patch'/'pixel'/'sample' for fixed sampled pixels, "
            "or 'full' for full-image validation."
        )

    if FORCE_TRAIN_WORKERS_ZERO is None:
        FORCE_TRAIN_WORKERS_ZERO = bool(DETERMINISTIC)

    seed_everything(SEED, deterministic=DETERMINISTIC, allow_tf32=ALLOW_TF32)

    train_num_workers = 0 if FORCE_TRAIN_WORKERS_ZERO else 4
    val_num_workers = 0
    test_num_workers = 0

    pin_memory = True
    persistent_workers = train_num_workers > 0
    prefetch_factor = None

    data_generator = torch.Generator()
    data_generator.manual_seed(SEED)
    worker_init_fn = make_worker_init_fn(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "This script requires a CUDA GPU."

    print(f"[device] {device}")
    print(f"[seed] {SEED}")
    print(f"[validation] mode={VALIDATION_MODE}")
    print(f"[precision] USE_AMP={USE_AMP}, ALLOW_TF32={ALLOW_TF32}")
    print(f"[determinism] DETERMINISTIC={DETERMINISTIC}, train_num_workers={train_num_workers}")

    def miou_accuracy(pred, gt, ncls):
        cm = np.bincount(ncls * gt + pred, minlength=ncls * ncls).reshape(ncls, ncls)
        tp = np.diag(cm)
        den = cm.sum(1) + cm.sum(0) - tp
        iou = tp / np.maximum(den, 1)
        accuracy = (pred == gt).mean()
        return float(np.nanmean(iou)), iou, float(accuracy)

    counts_classes = np.zeros(NCLASSES, dtype=np.int64)
    for i in train_ids:
        y = label_cache[i].ravel()
        for c in range(NCLASSES):
            counts_classes[c] += (y == c).sum()

    weights = counts_classes.sum() / (len(counts_classes) * np.maximum(counts_classes, 1))
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    print("[init] loading data indices...")
    print("  train:", train_ids)
    print("  val  :", val_ids)
    print("  test :", test_ids)
    print("[class counts]", counts_classes)
    print("[class weights]", np.round(weights, 6))

    class PixelDS(Dataset):
        def __init__(self, ids, nsamples):
            self.X = [data_cache[i] for i in ids]
            self.Y = [label_cache[i] for i in ids]
            self.ids = ids
            self.nsamples = nsamples

            H, W, K = self.X[0].shape
            print(f"[dataset-train] {len(ids)} files, sample shape: (H={H}, W={W}, K={K})")

            for idx, x, y in zip(ids, self.X, self.Y):
                if x.ndim != 3:
                    raise ValueError(f"Image {idx} must have shape (H, W, K), got {x.shape}")
                if y.shape != x.shape[:2]:
                    raise ValueError(
                        f"Label {idx} must have shape {x.shape[:2]}, got {y.shape}"
                    )

        def __len__(self):
            return self.nsamples

        def __getitem__(self, _):
            idx = random.randrange(len(self.X))
            x = self.X[idx]
            y = self.Y[idx]

            H, W, _ = x.shape
            i = random.randrange(H)
            j = random.randrange(W)

            xp = x[i, j, :]
            yp = y[i, j]

            xp = torch.from_numpy(np.asarray(xp, dtype=np.float32))
            yp = torch.tensor(int(yp), dtype=torch.long)
            return xp, yp

    class FixedPixelDS(Dataset):
        def __init__(self, ids, nsamples, seed):
            self.X = [data_cache[i] for i in ids]
            self.Y = [label_cache[i] for i in ids]
            self.ids = ids
            self.nsamples = nsamples

            H, W, K = self.X[0].shape
            print(f"[dataset-val-patch] {len(ids)} files, sample shape: (H={H}, W={W}, K={K})")

            for idx, x, y in zip(ids, self.X, self.Y):
                if x.ndim != 3:
                    raise ValueError(f"Image {idx} must have shape (H, W, K), got {x.shape}")
                if y.shape != x.shape[:2]:
                    raise ValueError(
                        f"Label {idx} must have shape {x.shape[:2]}, got {y.shape}"
                    )

            rng = np.random.default_rng(seed)
            self.samples = []

            for _ in range(nsamples):
                idx = int(rng.integers(len(self.X)))
                x = self.X[idx]
                H, W, _ = x.shape
                i = int(rng.integers(H))
                j = int(rng.integers(W))
                self.samples.append((idx, i, j))

        def __len__(self):
            return self.nsamples

        def __getitem__(self, index):
            idx, i, j = self.samples[index]
            x = self.X[idx]
            y = self.Y[idx]

            xp = x[i, j, :]
            yp = y[i, j]

            xp = torch.from_numpy(np.asarray(xp, dtype=np.float32))
            yp = torch.tensor(int(yp), dtype=torch.long)
            return xp, yp

    class FullPixelDS(Dataset):
        def __init__(self, ids, name="full"):
            self.ids = list(ids)
            self.offsets = []
            self.total_pixels = 0
            self.feature_dim = None

            for idx in self.ids:
                x = data_cache[idx]
                y = label_cache[idx]

                if x.ndim != 3:
                    raise ValueError(f"Image {idx} must have shape (H, W, K), got {x.shape}")
                if y.shape != x.shape[:2]:
                    raise ValueError(
                        f"Label {idx} must have shape {x.shape[:2]}, got {y.shape}"
                    )

                H, W, K = x.shape

                if self.feature_dim is None:
                    self.feature_dim = K
                elif K != self.feature_dim:
                    raise ValueError(
                        f"Image {idx} has K={K}, but previous images have K={self.feature_dim}"
                    )

                self.offsets.append((self.total_pixels, idx, H, W))
                self.total_pixels += H * W

            print(f"[dataset-{name}] total pixels: {self.total_pixels}, feature dim: {self.feature_dim}")

        def __len__(self):
            return self.total_pixels

        def __getitem__(self, index):
            for start, idx, H, W in reversed(self.offsets):
                if index >= start:
                    local = index - start
                    i = local // W
                    j = local % W
                    break
            else:
                raise IndexError(index)

            x = data_cache[idx]
            y = label_cache[idx]

            xp = x[i, j, :]
            yp = y[i, j]

            xp = torch.from_numpy(np.asarray(xp, dtype=np.float32))
            yp = torch.tensor(int(yp), dtype=torch.long)
            return xp, yp

    train_dl = DataLoader(
        PixelDS(train_ids, SAMPLES_PER_EPOCH_TRAIN),
        batch_size=BATCH,
        shuffle=True,
        num_workers=train_num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        generator=data_generator,
        worker_init_fn=worker_init_fn if train_num_workers > 0 else None,
    )

    if VALIDATION_MODE == "patch":
        val_dl = DataLoader(
            FixedPixelDS(val_ids, SAMPLES_PER_EPOCH_VAL, seed=SEED + 1),
            batch_size=BATCH,
            shuffle=False,
            num_workers=val_num_workers,
            pin_memory=(device.type == "cuda"),
        )
    else:
        val_dl = None

    K = data_cache[train_ids[0]].shape[-1]
    print(f"Shape of image: {data_cache[train_ids[0]].shape}")

    model = build_model(model_name, in_channels=K, n_classes=NCLASSES).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"[model] {model_name}(in_features={K}, ncls={NCLASSES}), params={params/1e3:.3f}k")
    print(f"[train] device={device}, BATCH={BATCH}, EPOCHS={EPOCHS}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    lossfn = nn.CrossEntropyLoss(weight=class_weights)
    lossfn_val = nn.CrossEntropyLoss(weight=class_weights, reduction="sum")
    lossfn_test = nn.CrossEntropyLoss(weight=class_weights, reduction="sum")

    def evaluate_full_pixels(model, ids, lossfn_sum, name="full"):
        full_dl = DataLoader(
            FullPixelDS(ids, name=name),
            batch_size=BATCH,
            shuffle=False,
            num_workers=test_num_workers,
            pin_memory=(device.type == "cuda"),
        )

        model.eval()

        loss_total = 0.0
        n_samples = 0
        total_pred = []
        total_gt = []

        with torch.no_grad():
            for xb, yb in full_dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                # Deployment-like full-image evaluation for the 1D model:
                # each pixel is classified independently in FP32, with no AMP.
                logits = model(xb)
                loss = lossfn_sum(logits, yb)

                loss_total += loss.item()
                n_samples += yb.size(0)

                pred = logits.argmax(dim=1).cpu().numpy()
                gt = yb.cpu().numpy()

                total_pred.append(pred)
                total_gt.append(gt)

        pred = np.concatenate(total_pred)
        gt = np.concatenate(total_gt)

        miou, iou_per_class, accuracy = miou_accuracy(pred, gt, ncls=NCLASSES)
        loss_mean = loss_total / n_samples

        return loss_mean, accuracy, miou, iou_per_class

    best_model = {
        "miou": -1.0,
        "epoch": 0,
        "state": None,
        "iou_per_class": None,
    }

    loss_train_list = []
    accuracy_train_list = []
    loss_val_list = []
    accuracy_val_list = []

    scaler = GradScaler(device="cuda", enabled=USE_AMP)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()

        running_loss = 0.0
        n_train_correct = 0
        n_train_samples = 0

        for xb, yb in train_dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with autocast("cuda", dtype=torch.float16, enabled=USE_AMP):
                logits = model(xb)
                loss = lossfn(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running_loss += loss.item() * yb.size(0)
            n_train_samples += yb.size(0)
            n_train_correct += (logits.argmax(1) == yb).sum().item()

        loss_train_list.append(running_loss / n_train_samples)
        accuracy_train_list.append(n_train_correct / n_train_samples)

        model.eval()

        if VALIDATION_MODE == "patch":
            loss_val = 0.0
            n_val_samples = 0
            total_pred, total_gt = [], []

            with torch.no_grad():
                for xb, yb in val_dl:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)

                    with autocast("cuda", dtype=torch.float16, enabled=USE_AMP):
                        logits = model(xb)
                        loss = lossfn_val(logits, yb)

                    loss_val += loss.item()
                    n_val_samples += yb.size(0)

                    pred = logits.argmax(1).cpu().numpy()
                    gt = yb.cpu().numpy()

                    total_pred.append(pred)
                    total_gt.append(gt)

            pred = np.concatenate(total_pred)
            gt = np.concatenate(total_gt)

            miou, iou_per_class, accuracy = miou_accuracy(pred, gt, ncls=NCLASSES)
            loss_val_mean = loss_val / n_val_samples

        else:
            loss_val_mean, accuracy, miou, iou_per_class = evaluate_full_pixels(
                model=model,
                ids=val_ids,
                lossfn_sum=lossfn_val,
                name="val-full",
            )

        loss_val_list.append(loss_val_mean)
        accuracy_val_list.append(accuracy)

        sched.step()

        dt = time.time() - t0
        print(
            f"Epoch: {epoch:02d} "
            f"Acc (Val, Train): ({accuracy:.4f},{(n_train_correct / n_train_samples):.4f})  "
            f"mIoU: {miou:.3f}  "
            f"IoU[Cloud, Land, Sea]: {np.round(iou_per_class, 3)}  "
            f"val_loss: {loss_val_mean:.6f}  "
            f"lr: {opt.param_groups[0]['lr']:.6e}  "
            f"time: {dt:.1f}s"
        )

        if epoch >= SAVE_FROM_EPOCH and miou > best_model["miou"]:
            best_model["miou"] = miou
            best_model["epoch"] = epoch
            best_model["state"] = copy.deepcopy(model.state_dict())
            best_model["iou_per_class"] = iou_per_class.copy()
            print(f"  [best] new best mIoU {miou:.3f} at epoch {epoch}")

    print("\nTraining complete.")
    print(
        f"Best mIoU: {best_model['miou']:.3f} at epoch {best_model['epoch']}  "
        f"IoU/cls: {np.round(best_model['iou_per_class'], 3)}"
    )

    if best_model["state"] is not None:
        model.load_state_dict(best_model["state"])

    model.eval()

    loss_test, accuracy_test, miou_test, iou_per_class_test = evaluate_full_pixels(
        model=model,
        ids=test_ids,
        lossfn_sum=lossfn_test,
        name="test",
    )

    print(
        f"[test] Acc: {accuracy_test:.4f}  "
        f"mIoU: {miou_test:.3f}  "
        f"IoU[Cloud, Land, Sea]: {np.round(iou_per_class_test, 3)}  "
        f"Loss: {loss_test:.6f}"
    )

    tag = f"pixel1d_e{best_model['epoch']}_miou{best_model['miou']:.3f}_K{K}".replace(".", "-")
    ckpt_path = out / f"{model_name}_{tag}.pt"

    raw_ckpt = {
        "model": model_name,
        "dataset": dataset_name,
        "loss_val_list": loss_val_list,
        "accuracy_val_list": accuracy_val_list,
        "loss_test": loss_test,
        "accuracy_test": accuracy_test,
        "miou_test": miou_test,
        "iou_per_class_test": iou_per_class_test,
        "loss_train_list": loss_train_list,
        "accuracy_train_list": accuracy_train_list,
        "state_dict": best_model["state"],
        "epoch": best_model["epoch"],
        "miou": best_model["miou"],
        "iou_per_class": best_model["iou_per_class"],
        "config": {
            "BATCH": BATCH,
            "EPOCHS": EPOCHS,
            "NCLASSES": NCLASSES,
            "LR": LR,
            "WEIGHT_DECAY": WEIGHT_DECAY,
            "SEED": SEED,
            "SAMPLES_PER_EPOCH_TRAIN": SAMPLES_PER_EPOCH_TRAIN,
            "SAMPLES_PER_EPOCH_VAL": SAMPLES_PER_EPOCH_VAL,
            "VALIDATION_MODE": VALIDATION_MODE,
            "USE_AMP": USE_AMP,
            "DETERMINISTIC": DETERMINISTIC,
            "ALLOW_TF32": ALLOW_TF32,
            "FORCE_TRAIN_WORKERS_ZERO": FORCE_TRAIN_WORKERS_ZERO,
            "FULL_IMAGE_EVAL_POLICY": "classify_each_pixel_once_fp32",
            "train_ids": train_ids,
            "val_ids": val_ids,
            "test_ids": test_ids,
        },
    }

    clean_ckpt = to_python(raw_ckpt)
    torch.save(clean_ckpt, ckpt_path)
    print(f"[save] saved model to: {ckpt_path}")

    return miou_test, iou_per_class_test, accuracy_test
