from pathlib import Path
import gc
import numpy as np
from train_1d_fold import train_1d_fold
from train_2d_fold import train_2d_fold



def load_data_cache(base: Path, ids: list[int], mmap_mode=None):
    data_cache = {}
    label_cache = {}
    print(f"[cache] Loading {len(ids)} files (mmap_mode={mmap_mode})...")

    for i in ids:
        data_cache[i] = np.load(base / f"data{i}.npy", mmap_mode=mmap_mode)
        if data_cache[i].dtype != np.float32:
            data_cache[i] = data_cache[i].astype(np.float32)

        label_cache[i] = np.load(base / f"label{i}.npy", mmap_mode=mmap_mode)
        if label_cache[i].dtype != np.int64:
            label_cache[i] = label_cache[i].astype(np.int64)

    return data_cache, label_cache


def infer_spectral_axis(data: np.ndarray, label: np.ndarray) -> int:
    """Infer the spectral/channel axis by matching the remaining axes to label shape."""
    if data.ndim != label.ndim + 1:
        raise ValueError(
            f"Expected data.ndim == label.ndim + 1, got data shape {data.shape} "
            f"and label shape {label.shape}"
        )

    for axis in range(data.ndim):
        spatial_shape = data.shape[:axis] + data.shape[axis + 1:]
        if spatial_shape == label.shape:
            return axis

    raise ValueError(
        f"Could not infer spectral axis from data shape {data.shape} "
        f"and label shape {label.shape}"
    )


def calculate_zscore_stats(
    data_cache: dict[int, np.ndarray],
    label_cache: dict[int, np.ndarray],
    train_ids: list[int],
    eps: float = 1e-6,
    min_std: float = 1e-3,
):
    """Calculate per-band mean/std using only training images for one fold."""
    if not train_ids:
        raise ValueError("train_ids must not be empty")

    spectral_axis = infer_spectral_axis(data_cache[train_ids[0]], label_cache[train_ids[0]])
    n_bands = data_cache[train_ids[0]].shape[spectral_axis]

    total = np.zeros(n_bands, dtype=np.float64)
    total_sq = np.zeros(n_bands, dtype=np.float64)
    n_values = 0

    for i in train_ids:
        axis_i = infer_spectral_axis(data_cache[i], label_cache[i])
        if axis_i != spectral_axis:
            raise ValueError(
                f"Inconsistent spectral axis for image {i}: expected {spectral_axis}, got {axis_i}"
            )

        if data_cache[i].shape[spectral_axis] != n_bands:
            raise ValueError(
                f"Inconsistent band count for image {i}: expected {n_bands}, "
                f"got {data_cache[i].shape[spectral_axis]}"
            )

        x = np.moveaxis(data_cache[i], spectral_axis, -1).reshape(-1, n_bands)
        x = x.astype(np.float64, copy=False)
        total += x.sum(axis=0)
        total_sq += np.square(x).sum(axis=0)
        n_values += x.shape[0]

    if n_values == 0:
        raise ValueError("No training pixels found while calculating z-score statistics")

    mean = total / n_values
    var = (total_sq / n_values) - np.square(mean)
    var = np.maximum(var, eps * eps)
    std = np.sqrt(var)
    std = np.maximum(std, min_std)

    return mean.astype(np.float32), std.astype(np.float32), spectral_axis


def make_zscore_data_cache(
    data_cache: dict[int, np.ndarray],
    ids: list[int],
    mean: np.ndarray,
    std: np.ndarray,
    spectral_axis: int,
):
    """Apply fold-specific train statistics to train/validation/test images."""
    if not ids:
        return {}

    first_id = ids[0]
    reshape = [1] * data_cache[first_id].ndim
    reshape[spectral_axis] = -1
    mean_view = mean.reshape(reshape)
    std_view = std.reshape(reshape)

    zscore_cache = {}
    for i in ids:
        x = data_cache[i].astype(np.float32, copy=False)
        zscore_cache[i] = ((x - mean_view) / std_view).astype(np.float32, copy=False)

    return zscore_cache


def main():
    n_images = 60
    HARDCODED_FOLDS = [
        {
            "fold": 0,
            "val_ids": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            "test_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        },
        {
            "fold": 1,
            "val_ids": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            "test_ids": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        },
        {
            "fold": 2,
            "val_ids": [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
            "test_ids": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        },
        {
            "fold": 3,
            "val_ids": [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
            "test_ids": [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
        },
        {
            "fold": 4,
            "val_ids": [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
            "test_ids": [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
        },
        {
            "fold": 5,
            "val_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "test_ids": [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
        },
    ]

    all_ids = list(range(n_images))
    folds = []
    for fold in HARDCODED_FOLDS:
        val_ids = fold["val_ids"]
        test_ids = fold["test_ids"]

        train_ids = [
            i for i in all_ids
            if i not in val_ids and i not in test_ids
        ]

        folds.append({
            "fold": fold["fold"],
            "train_ids": train_ids,
            "val_ids": val_ids,
            "test_ids": test_ids,
        })

    for fold in folds:
        train_ids = fold["train_ids"]
        val_ids = fold["val_ids"]
        test_ids = fold["test_ids"]

        assert len(set(train_ids) & set(val_ids)) == 0
        assert len(set(train_ids) & set(test_ids)) == 0
        assert len(set(val_ids) & set(test_ids)) == 0

        combined = sorted(train_ids + val_ids + test_ids)
        assert combined == list(range(n_images))

    print("[folds] Hardcoded folds validated.")

    run_i = "001_seed42_strict_fp32_zscore"
    seed = 42
    PATCH_SIZE = 32
    dataset_name = "dataset_name"
    dataset_short = f"{Path(dataset_name).name}_P{PATCH_SIZE}"
    base = Path("/home/datasets") / dataset_name

    data_cache, label_cache = load_data_cache(
        base=base,
        ids=all_ids,
        mmap_mode=None,
    )

    results: dict[str, dict[str, list]] = {}

    def store_result(result_name, miou, iou, acc):
        if result_name not in results:
            results[result_name] = {"acc": [], "miou": [], "iou": []}

        results[result_name]["acc"].append(acc)
        results[result_name]["miou"].append(miou)
        results[result_name]["iou"].append(iou)

    models_1d = [
        ("justoliunet", "justoliunet"),
        ("justoliunet_bn", "justoliunet_bn"),
        ("spectralpointwise1D", "spectralpointwise1D"),
    ]

    models_2d = [
        ("sp_unet_large", "sp_unet_large"),
        ("sp_unet_medium", "sp_unet_medium"),
        ("sp_unet_small", "sp_unet_small"),
        ("waveletcnn", "waveletcnn"),
        ("justounetsimple", "justounetsimple"),
    ]

    for fold in folds:
        fold_id = fold["fold"]
        train_ids = fold["train_ids"]
        val_ids = fold["val_ids"]
        test_ids = fold["test_ids"]
        fold_ids = sorted(train_ids + val_ids + test_ids)

        print(f"\n===== FOLD {fold_id} =====")
        print(f"Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

        mean, std, spectral_axis = calculate_zscore_stats(
            data_cache=data_cache,
            label_cache=label_cache,
            train_ids=train_ids,
        )

        stats_dir = Path(f"model_weights_{run_i}") / dataset_short / "zscore_stats" / f"fold_{fold_id}"
        stats_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            stats_dir / "stats.npz",
            mean=mean,
            std=std,
            spectral_axis=np.array(spectral_axis, dtype=np.int64),
            train_ids=np.array(train_ids, dtype=np.int64),
            val_ids=np.array(val_ids, dtype=np.int64),
            test_ids=np.array(test_ids, dtype=np.int64),
        )

        print(
            f"[zscore] Fold {fold_id}: train-only stats, "
            f"spectral_axis={spectral_axis}, bands={mean.shape[0]}"
        )

        zscore_data_cache = make_zscore_data_cache(
            data_cache=data_cache,
            ids=fold_ids,
            mean=mean,
            std=std,
            spectral_axis=spectral_axis,
        )

        def run_and_store_1d(model_name, out_name):
            out_fold = Path(f"model_weights_{run_i}") / dataset_short / out_name / f"fold_{fold_id}"

            miou, iou, acc = train_1d_fold(
                model_name=model_name,
                dataset_name=dataset_name,
                base=base,
                out=out_fold,
                train_ids=train_ids,
                val_ids=val_ids,
                test_ids=test_ids,
                SAVE_FROM_EPOCH=20,
                BATCH=4096,
                EPOCHS=80,
                NCLASSES=3,
                LR=5e-4,
                WEIGHT_DECAY=1e-3,
                SAMPLES_PER_EPOCH_TRAIN=300000,
                SAMPLES_PER_EPOCH_VAL=300000,
                SEED=seed,
                data_cache=zscore_data_cache,
                label_cache=label_cache,
                VALIDATION_MODE="sample",      # "patch"/"pixel"/"sample" or "full"
                USE_AMP=False,                  # mixed precision training/patch validation for speed
                DETERMINISTIC=True,             # more repeatable runs, usually slower
                ALLOW_TF32=False,               # faster on Ampere+, less bitwise-stable
                FORCE_TRAIN_WORKERS_ZERO=None,  # None -> True when deterministic, else False
            )

            store_result(model_name, miou, iou, acc)

        def run_and_store_2d(model_name, out_name):
            out_fold = Path(f"model_weights_{run_i}") / dataset_short / out_name / f"fold_{fold_id}"

            miou, iou, acc = train_2d_fold(
                model_name=model_name,
                dataset_name=dataset_name,
                base=base,
                out=out_fold,
                train_ids=train_ids,
                val_ids=val_ids,
                test_ids=test_ids,
                PATCH_SIZE=PATCH_SIZE,
                SAVE_FROM_EPOCH=0,
                BATCH=64,
                EPOCHS=80,
                NCLASSES=3,
                LR=1e-3,
                WEIGHT_DECAY=1e-4,
                SAMPLES_PER_EPOCH_TRAIN=2000,
                SAMPLES_PER_EPOCH_VAL=2000,
                SEED=seed,
                data_cache=zscore_data_cache,
                label_cache=label_cache,
                VALIDATION_MODE="full",        # "patch" or "full"
                USE_AMP=False,                  # mixed precision training/validation for speed
                DETERMINISTIC=True,             # more repeatable runs, usually slower
                ALLOW_TF32=False,               # faster on Ampere+, less bitwise-stable
                FORCE_TRAIN_WORKERS_ZERO=None,  # None -> True when deterministic, else False
            )

            store_result(model_name, miou, iou, acc)

            store_result(model_name, miou, iou, acc)

        # ---------- 1D models ----------
        for model_name, out_name in models_1d:
            run_and_store_1d(model_name, out_name)


        # ---------- 2D patch models ----------
        for model_name, out_name in models_2d:
            run_and_store_2d(model_name, out_name)

        del zscore_data_cache
        gc.collect()

    out_file = Path(f"model_weights_{run_i}") / dataset_short / "results.txt"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    def summarize(name, acc, miou):
        acc_mean, acc_std = np.mean(acc), np.std(acc)
        miou_mean, miou_std = np.mean(miou), np.std(miou)

        return (
            f"{name}\n"
            f"  Acc  : {acc_mean:.4f} ± {acc_std:.4f}\n"
            f"  mIoU : {miou_mean:.4f} ± {miou_std:.4f}\n"
            f"  Acc per fold  : {np.round(acc, 4)}\n"
            f"  mIoU per fold : {np.round(miou, 4)}\n"
        )

    with open(out_file, "w") as f:
        f.write("==== Cross-Validation Results, z-score only ====\n")
        f.write("Per-band z-score uses train-fold statistics only.\n\n")

        for model_name, metric_dict in results.items():
            f.write(
                summarize(
                    model_name,
                    metric_dict["acc"],
                    metric_dict["miou"],
                ) + "\n"
            )

    print(f"[results] Saved to {out_file}")


if __name__ == "__main__":
    main()
