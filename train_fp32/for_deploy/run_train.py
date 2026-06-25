from pathlib import Path
import numpy as np

from train_1d_fold import train_1d_fold
from train_2d_fold import train_2d_fold


VAL_IDS = [0, 1, 5, 35, 37, 41, 47, 52]
TEST_IDS = [12, 16, 17, 23, 24, 46, 54, 58]


def build_deployment_split(n_images: int):
    all_ids = list(range(n_images))

    val_ids = list(VAL_IDS)
    test_ids = list(TEST_IDS)

    val_set = set(val_ids)
    test_set = set(test_ids)

    assert len(val_ids) == len(val_set), "Duplicate IDs found in VAL_IDS"
    assert len(test_ids) == len(test_set), "Duplicate IDs found in TEST_IDS"
    assert len(val_set & test_set) == 0, "VAL_IDS and TEST_IDS overlap"

    invalid_ids = sorted((val_set | test_set) - set(all_ids))
    assert not invalid_ids, f"IDs outside range 0..{n_images - 1}: {invalid_ids}"

    train_ids = [
        i for i in all_ids
        if i not in val_set and i not in test_set
    ]

    assert len(set(train_ids) & val_set) == 0
    assert len(set(train_ids) & test_set) == 0
    assert len(val_set & test_set) == 0

    combined = sorted(train_ids + val_ids + test_ids)
    assert combined == all_ids

    return all_ids, train_ids, val_ids, test_ids


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


def main():
    n_images = 60

    all_ids, train_ids, val_ids, test_ids = build_deployment_split(n_images)

    print("[split] Fixed deployment split validated.")
    print(f"Train IDs ({len(train_ids)}): {train_ids}")
    print(f"Val IDs   ({len(val_ids)}): {val_ids}")
    print(f"Test IDs  ({len(test_ids)}): {test_ids}")

    run_i = "deploy_seed42_strict_fp32"
    seed = 42

    PATCH_SIZE = 32
    NCLASSES = 3

    dataset_name = "dataset_name"
    dataset_short = f"{Path(dataset_name).name}_P{PATCH_SIZE}"
    base = Path("/home/datasets") / dataset_name

    output_root = Path(f"model_weights_{run_i}") / dataset_short

    data_cache, label_cache = load_data_cache(
        base=base,
        ids=all_ids,
        mmap_mode=None,
    )

    results = {}

    def store_result(model_name, miou, iou, acc):
        results[model_name] = {
            "acc": acc,
            "miou": miou,
            "iou": iou,
        }

    def run_and_store_1d(model_name, out_name):
        out_dir = output_root / out_name / "deploy_split"

        miou, iou, acc = train_1d_fold(
            model_name=model_name,
            dataset_name=dataset_name,
            base=base,
            out=out_dir,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            SAVE_FROM_EPOCH=20,
            BATCH=4096,
            EPOCHS=80,
            NCLASSES=NCLASSES,
            LR=5e-4,
            WEIGHT_DECAY=1e-3,
            SAMPLES_PER_EPOCH_TRAIN=300000,
            SAMPLES_PER_EPOCH_VAL=300000,
            SEED=seed,
            data_cache=data_cache,
            label_cache=label_cache,
            VALIDATION_MODE="sample",
            USE_AMP=False,
            DETERMINISTIC=True,
            ALLOW_TF32=False,
            FORCE_TRAIN_WORKERS_ZERO=None,
        )

        store_result(model_name, miou, iou, acc)

    def run_and_store_2d(model_name, out_name):
        out_dir = output_root / out_name / "deploy_split"

        miou, iou, acc = train_2d_fold(
            model_name=model_name,
            dataset_name=dataset_name,
            base=base,
            out=out_dir,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            PATCH_SIZE=PATCH_SIZE,
            SAVE_FROM_EPOCH=0,
            BATCH=64,
            EPOCHS=80,
            NCLASSES=NCLASSES,
            LR=1e-3,
            WEIGHT_DECAY=1e-4,
            SAMPLES_PER_EPOCH_TRAIN=2000,
            SAMPLES_PER_EPOCH_VAL=2000,
            SEED=seed,
            data_cache=data_cache,
            label_cache=label_cache,
            VALIDATION_MODE="full",
            USE_AMP=False,
            DETERMINISTIC=True,
            ALLOW_TF32=False,
            FORCE_TRAIN_WORKERS_ZERO=None,
        )

        store_result(model_name, miou, iou, acc)


        store_result(model_name, miou, iou, acc)

    print("\n===== DEPLOYMENT TRAINING SPLIT =====")
    print(f"Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

    # ---------- 1D models ----------
    run_and_store_1d("justoliunet", "justoliunet")
    run_and_store_1d("justoliunet_bn", "justoliunet_bn")
    run_and_store_1d("spectralpointwise1D", "spectralpointwise1D")

    run_and_store_2d("sp_unet_large", "sp_unet_large")
    run_and_store_2d("sp_unet_medium", "sp_unet_medium")
    run_and_store_2d("sp_unet_small", "sp_unet_small")
    run_and_store_2d("waveletcnn", "waveletcnn")
    run_and_store_2d("justounetsimple", "justounetsimple")

    out_file = output_root / "deployment_results.txt"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w") as f:
        f.write("==== Fixed Deployment Split Results ====\n\n")

        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Patch size: {PATCH_SIZE}\n")
        f.write(f"Seed: {seed}\n\n")

        f.write(f"Train IDs ({len(train_ids)}): {train_ids}\n")
        f.write(f"Val IDs   ({len(val_ids)}): {val_ids}\n")
        f.write(f"Test IDs  ({len(test_ids)}): {test_ids}\n\n")

        for model_name, metrics in results.items():
            acc = metrics["acc"]
            miou = metrics["miou"]
            iou = metrics["iou"]

            f.write(f"{model_name}\n")
            f.write(f"  Acc  : {acc:.4f}\n")
            f.write(f"  mIoU : {miou:.4f}\n")
            f.write(f"  IoU  : {np.round(iou, 4)}\n\n")

    print(f"[results] Saved to {out_file}")


if __name__ == "__main__":
    main()