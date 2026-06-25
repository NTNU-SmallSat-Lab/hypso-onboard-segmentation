from pathlib import Path
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


def train_2d_fold(
    model_name: str,
    dataset_name: str,
    base: Path,
    out: Path,
    train_ids: list,
    val_ids: list,
    test_ids: list,
    PATCH_SIZE: int,
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
    VALIDATION_MODE: str = "full",      # "patch" or "full"
    USE_AMP: bool = False,                 # mixed precision training/validation for speed
    DETERMINISTIC: bool = True,          # more repeatable runs, usually slower
    ALLOW_TF32: bool = True,              # faster on Ampere+, less bitwise-stable
    FORCE_TRAIN_WORKERS_ZERO: bool = None # None -> True when deterministic, else False
):
    base = Path(base)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    VALIDATION_MODE = str(VALIDATION_MODE).lower()
    if VALIDATION_MODE not in {"patch", "full"}:
        raise ValueError("VALIDATION_MODE must be either 'patch' or 'full'.")

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
    print(f"[patch] PATCH_SIZE={PATCH_SIZE}")
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

    class PatchDS(Dataset):
        def __init__(self, ids, nsamples):
            self.X = [data_cache[i] for i in ids]
            self.Y = [label_cache[i] for i in ids]
            self.ids = ids
            self.nsamples = nsamples

            H, W, K = self.X[0].shape
            print(f"[dataset-train] {len(ids)} files, sample shape: (H={H}, W={W}, C={K})")

            for idx, x in zip(ids, self.X):
                H, W, _ = x.shape
                if H < PATCH_SIZE or W < PATCH_SIZE:
                    raise ValueError(
                        f"Image {idx} is smaller than PATCH_SIZE={PATCH_SIZE}: shape={x.shape}"
                    )

        def __len__(self):
            return self.nsamples

        def __getitem__(self, _):
            # Random image.
            idx = random.randrange(len(self.X))
            x = self.X[idx]
            y = self.Y[idx]

            H, W, _ = x.shape

            # Random valid top-left corner.
            i0 = random.randrange(0, H - PATCH_SIZE + 1)
            j0 = random.randrange(0, W - PATCH_SIZE + 1)

            xp = x[i0:i0 + PATCH_SIZE, j0:j0 + PATCH_SIZE, :]
            yp = y[i0:i0 + PATCH_SIZE, j0:j0 + PATCH_SIZE]

            # HWC -> CHW for PyTorch Conv2D.
            xp = np.transpose(xp, (2, 0, 1))

            xp = torch.from_numpy(np.asarray(xp, dtype=np.float32))
            yp = torch.from_numpy(np.asarray(yp, dtype=np.int64))

            return xp, yp

    class FixedPatchDS(Dataset):
        def __init__(self, ids, nsamples, seed):
            self.X = [data_cache[i] for i in ids]
            self.Y = [label_cache[i] for i in ids]
            self.ids = ids
            self.nsamples = nsamples

            H, W, K = self.X[0].shape
            print(f"[dataset-val-patch] {len(ids)} files, sample shape: (H={H}, W={W}, C={K})")

            for idx, x in zip(ids, self.X):
                H, W, _ = x.shape
                if H < PATCH_SIZE or W < PATCH_SIZE:
                    raise ValueError(
                        f"Image {idx} is smaller than PATCH_SIZE={PATCH_SIZE}: shape={x.shape}"
                    )

            rng = np.random.default_rng(seed)
            self.samples = []

            for _ in range(nsamples):
                idx = int(rng.integers(len(self.X)))
                x = self.X[idx]

                H, W, _ = x.shape

                # Validation samples are random once, then reused every epoch.
                i0 = int(rng.integers(0, H - PATCH_SIZE + 1))
                j0 = int(rng.integers(0, W - PATCH_SIZE + 1))

                self.samples.append((idx, i0, j0))

        def __len__(self):
            return self.nsamples

        def __getitem__(self, index):
            idx, i0, j0 = self.samples[index]

            x = self.X[idx]
            y = self.Y[idx]

            xp = x[i0:i0 + PATCH_SIZE, j0:j0 + PATCH_SIZE, :]
            yp = y[i0:i0 + PATCH_SIZE, j0:j0 + PATCH_SIZE]

            # HWC -> CHW for PyTorch Conv2D.
            xp = np.transpose(xp, (2, 0, 1))

            xp = torch.from_numpy(np.asarray(xp, dtype=np.float32))
            yp = torch.from_numpy(np.asarray(yp, dtype=np.int64))

            return xp, yp

    class FullCoverPatchDS(Dataset):
        def __init__(self, ids, name="full"):
            self.samples = []

            def starts(L):
                s = list(range(0, L - PATCH_SIZE + 1, PATCH_SIZE))
                last = L - PATCH_SIZE

                # Add the final edge patch only when the image size is not divisible
                # by PATCH_SIZE. This may create overlap at the bottom/right edges.
                if s[-1] != last:
                    s.append(last)

                return s

            for idx in ids:
                H, W, C = data_cache[idx].shape

                if H < PATCH_SIZE or W < PATCH_SIZE:
                    raise ValueError(f"Image {idx} is smaller than PATCH_SIZE={PATCH_SIZE}")

                for i0 in starts(H):
                    for j0 in starts(W):
                        self.samples.append((idx, i0, j0))

            #print(f"[dataset-{name}] edge-covering patches: {len(self.samples)}")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            idx, i0, j0 = self.samples[index]
            x = data_cache[idx]

            xp = x[i0:i0 + PATCH_SIZE, j0:j0 + PATCH_SIZE, :]
            xp = np.transpose(xp, (2, 0, 1))
            xp = torch.from_numpy(np.asarray(xp, dtype=np.float32))

            return xp, idx, i0, j0

    train_dl = DataLoader(
        PatchDS(train_ids, SAMPLES_PER_EPOCH_TRAIN),
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
            FixedPatchDS(val_ids, SAMPLES_PER_EPOCH_VAL, seed=SEED + 1),
            batch_size=BATCH,
            shuffle=False,
            num_workers=val_num_workers,
            pin_memory=(device.type == "cuda"),
        )
    else:
        val_dl = None

    C = data_cache[train_ids[0]].shape[-1]
    print(f"Shape of image: {data_cache[train_ids[0]].shape}")

    model = build_model(model_name, in_channels=C, n_classes=NCLASSES).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"[model] {model_name}(in_channels={C}, ncls={NCLASSES}), params={params/1e3:.3f}k")
    print(f"[train] device={device}, BATCH={BATCH}, EPOCHS={EPOCHS}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    lossfn = nn.CrossEntropyLoss(weight=class_weights)
    lossfn_val = nn.CrossEntropyLoss(weight=class_weights, reduction="sum")
    lossfn_test = nn.CrossEntropyLoss(weight=class_weights, reduction="sum")

    def evaluate_full_images(model, ids, lossfn_sum, name="full"):
        patch_dl = DataLoader(
            FullCoverPatchDS(ids, name=name),
            batch_size=BATCH,
            shuffle=False,
            num_workers=test_num_workers,
            pin_memory=(device.type == "cuda"),
        )

        logits_full = {}
        filled = {}

        for idx in ids:
            H, W, _ = data_cache[idx].shape
            logits_full[idx] = torch.empty((NCLASSES, H, W), dtype=torch.float32)
            filled[idx] = torch.zeros((H, W), dtype=torch.bool)

        model.eval()

        with torch.no_grad():
            for xb, idxb, i0b, j0b in patch_dl:
                xb = xb.to(device, non_blocking=True)

                # Deployment-like full-image evaluation: no AMP and no overlap averaging.
                # Edge patches may overlap, but later overlapping pixels are discarded.
                logits = model(xb).cpu()

                for b in range(logits.shape[0]):
                    idx = int(idxb[b])
                    i0 = int(i0b[b])
                    j0 = int(j0b[b])

                    i1 = i0 + PATCH_SIZE
                    j1 = j0 + PATCH_SIZE

                    dest = logits_full[idx][:, i0:i1, j0:j1]
                    filled_region = filled[idx][i0:i1, j0:j1]

                    # First prediction wins.
                    # Pixels already predicted by earlier patches are not overwritten.
                    write_mask = ~filled_region

                    if write_mask.any():
                        dest[:, write_mask] = logits[b][:, write_mask]
                        filled_region[write_mask] = True

        loss_total = 0.0
        n_pixels = 0
        total_pred = []
        total_gt = []

        for idx in ids:
            if not filled[idx].all():
                missing = int((~filled[idx]).sum().item())
                raise RuntimeError(f"{name}: image {idx} has {missing} unfilled pixels.")

            y = label_cache[idx]
            gt_tensor = torch.from_numpy(np.asarray(y, dtype=np.int64))

            loss = lossfn_sum(
                logits_full[idx].unsqueeze(0).to(device),
                gt_tensor.unsqueeze(0).to(device),
            )

            loss_total += loss.item()
            n_pixels += gt_tensor.numel()

            pred = logits_full[idx].argmax(0).numpy().reshape(-1)
            gt = y.reshape(-1)

            total_pred.append(pred)
            total_gt.append(gt)

        pred = np.concatenate(total_pred)
        gt = np.concatenate(total_gt)

        miou, iou_per_class, accuracy = miou_accuracy(pred, gt, ncls=NCLASSES)
        loss_mean = loss_total / n_pixels

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
        n_train_pixels = 0

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

            pred = logits.argmax(1)
            running_loss += loss.item() * yb.numel()
            n_train_pixels += yb.numel()
            n_train_correct += (pred == yb).sum().item()

        loss_train_list.append(running_loss / n_train_pixels)
        accuracy_train_list.append(n_train_correct / n_train_pixels)

        model.eval()

        if VALIDATION_MODE == "patch":
            loss_val = 0.0
            n_val_pixels = 0
            total_pred, total_gt = [], []

            with torch.no_grad():
                for xb, yb in val_dl:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)

                    with autocast("cuda", dtype=torch.float16, enabled=USE_AMP):
                        logits = model(xb)
                        loss = lossfn_val(logits, yb)

                    loss_val += loss.item()
                    n_val_pixels += yb.numel()

                    pred = logits.argmax(1).cpu().numpy().reshape(-1)
                    gt = yb.cpu().numpy().reshape(-1)

                    total_pred.append(pred)
                    total_gt.append(gt)

            pred = np.concatenate(total_pred)
            gt = np.concatenate(total_gt)

            miou, iou_per_class, accuracy = miou_accuracy(pred, gt, ncls=NCLASSES)
            loss_val_mean = loss_val / n_val_pixels

        else:
            loss_val_mean, accuracy, miou, iou_per_class = evaluate_full_images(
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
            f"Acc (Val, Train): ({accuracy:.4f},{(n_train_correct / n_train_pixels):.4f})  "
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

    loss_test, accuracy_test, miou_test, iou_per_class_test = evaluate_full_images(
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

    tag = f"patch2d_e{best_model['epoch']}_miou{best_model['miou']:.3f}_C{C}_P{PATCH_SIZE}".replace(".", "-")
    ckpt_path = out / f"{model_name}_{tag}.pt"

    raw_ckpt = {
        "model": model_name,
        "dataset": dataset_name,
        "patch_size": PATCH_SIZE,
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
            "PATCH_SIZE": PATCH_SIZE,
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
            "FULL_IMAGE_OVERLAP_POLICY": "first_prediction_wins_discard_later_overlaps",
            "train_ids": train_ids,
            "val_ids": val_ids,
            "test_ids": test_ids,
        },
    }

    clean_ckpt = to_python(raw_ckpt)
    torch.save(clean_ckpt, ckpt_path)
    print(f"[save] saved model to: {ckpt_path}")

    return miou_test, iou_per_class_test, accuracy_test
