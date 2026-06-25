from pathlib import Path
import numpy as np
from train_1d_fold import train_1d_fold
from train_2d_fold import train_2d_fold



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

    results = {
        # 1D
        "justoliunet": {"acc": [], "miou": [], "iou": []},
        "justoliunet_bn": {"acc": [], "miou": [], "iou": []},
        "spectralpointwise1D": {"acc": [], "miou": [], "iou": []},
        

        # 2D patch models
        "cfcn": {"acc": [], "miou": [], "iou": []},
        "cunet": {"acc": [], "miou": [], "iou": []},
        "cunetpp": {"acc": [], "miou": [], "iou": []},
        "sp_unet_large": {"acc": [], "miou": [], "iou": []},
        "sp_unet_medium": {"acc": [], "miou": [], "iou": []},
        "sp_unet_small": {"acc": [], "miou": [], "iou": []},
        "waveletcnn": {"acc": [], "miou": [], "iou": []},
        "justounetsimple": {"acc": [], "miou": [], "iou": []},
    }

    run_i = "001_seed42_strict_fp32"
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


    for fold in folds:
        fold_id = fold["fold"]
        train_ids = fold["train_ids"]
        val_ids = fold["val_ids"]
        test_ids = fold["test_ids"]

        print(f"\n===== FOLD {fold_id} =====")
        print(f"Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

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
                LR= 5e-4,
                WEIGHT_DECAY= 1e-3,
                SAMPLES_PER_EPOCH_TRAIN=300000,
                SAMPLES_PER_EPOCH_VAL=300000,
                SEED=seed,
                data_cache=data_cache,
                label_cache=label_cache,
                VALIDATION_MODE = "sample",      # "patch"/"pixel"/"sample" or "full"
                USE_AMP = False,                 # mixed precision training/patch validation for speed
                DETERMINISTIC = True,          # more repeatable runs, usually slower
                ALLOW_TF32 = False,              # faster on Ampere+, less bitwise-stable
                FORCE_TRAIN_WORKERS_ZERO = None # None -> True when deterministic, else False
            )

            results[model_name]["acc"].append(acc)
            results[model_name]["miou"].append(miou)
            results[model_name]["iou"].append(iou)

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
                data_cache=data_cache,
                label_cache=label_cache,
                VALIDATION_MODE = "full",      # "patch" or "full"
                USE_AMP = False,                 # mixed precision training/validation for speed
                DETERMINISTIC = True,          # more repeatable runs, usually slower
                ALLOW_TF32 = False,              # faster on Ampere+, less bitwise-stable
                FORCE_TRAIN_WORKERS_ZERO = None # None -> True when deterministic, else False
            )

            results[model_name]["acc"].append(acc)
            results[model_name]["miou"].append(miou)
            results[model_name]["iou"].append(iou)

            results[model_name]["acc"].append(acc)
            results[model_name]["miou"].append(miou)
            results[model_name]["iou"].append(iou)

        # ---------- 1D models ----------
        run_and_store_1d("justoliunet", "justoliunet")
        run_and_store_1d("justoliunet_bn", "justoliunet_bn")
        run_and_store_1d("spectralpointwise1D", "spectralpointwise1D")

        run_and_store_2d("sp_unet_large", "sp_unet_large")
        run_and_store_2d("sp_unet_medium", "sp_unet_medium")
        run_and_store_2d("sp_unet_small", "sp_unet_small")
        run_and_store_2d("waveletcnn", "waveletcnn")
        run_and_store_2d("justounetsimple", "justounetsimple")

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
        f.write("==== Cross-Validation Results ====\n\n")

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