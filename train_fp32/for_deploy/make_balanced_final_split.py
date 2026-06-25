from pathlib import Path
import json
import numpy as np


DATASET_NAME = "dataset_name"
BASE = Path("/home/datasets") / DATASET_NAME

N_IMAGES = 60
NCLASSES = 3
CLASS_NAMES = ["Cloud", "Land", "Sea"]

N_VAL = 8
N_TEST = 8

SEED = 42
N_RANDOM_RESTARTS = 1000
MAX_GREEDY_SWAPS = 50

OUT_DIR = Path("final_splits") / DATASET_NAME.replace("/", "_")
OUT_PATH = OUT_DIR / "balanced_final_split.json"


def load_label_counts(base: Path, n_images: int, nclasses: int):
    counts = np.zeros((n_images, nclasses), dtype=np.int64)

    for i in range(n_images):
        label_path = base / f"label{i}.npy"
        y = np.load(label_path)

        if y.min() < 0 or y.max() >= nclasses:
            raise ValueError(
                f"Invalid label value in image {i}: "
                f"min={y.min()}, max={y.max()}, expected 0..{nclasses - 1}"
            )

        counts[i] = np.bincount(y.reshape(-1), minlength=nclasses)[:nclasses]

    return counts


def distribution_from_counts(counts):
    total = int(counts.sum())
    percent = counts / max(total, 1) * 100.0
    return counts.astype(int), percent, total


def score_counts(split_counts, global_percent):
    _, split_percent, _ = distribution_from_counts(split_counts)

    # Mean absolute percentage-point difference from global distribution.
    return float(np.mean(np.abs(split_percent - global_percent)))


def score_split(image_counts, train_ids, val_ids, test_ids, global_percent):
    train_counts = image_counts[train_ids].sum(axis=0)
    val_counts = image_counts[val_ids].sum(axis=0)
    test_counts = image_counts[test_ids].sum(axis=0)

    train_score = score_counts(train_counts, global_percent)
    val_score = score_counts(val_counts, global_percent)
    test_score = score_counts(test_counts, global_percent)

    # Val and test matter most because they are small.
    # Train gets a lower weight because it contains most images and is usually stable.
    total_score = (
        0.50 * train_score +
        1.00 * val_score +
        1.00 * test_score
    )

    return total_score, train_score, val_score, test_score


def make_ids_from_assignment(assignment):
    train_ids = np.where(assignment == 0)[0].tolist()
    val_ids = np.where(assignment == 1)[0].tolist()
    test_ids = np.where(assignment == 2)[0].tolist()
    return train_ids, val_ids, test_ids


def random_assignment(rng, n_images, n_val, n_test):
    ids = rng.permutation(n_images)

    assignment = np.zeros(n_images, dtype=np.int64)
    assignment[ids[:n_val]] = 1
    assignment[ids[n_val:n_val + n_test]] = 2

    return assignment


def greedy_improve_assignment(image_counts, assignment, global_percent, max_swaps):
    train_ids, val_ids, test_ids = make_ids_from_assignment(assignment)

    best_score, _, _, _ = score_split(
        image_counts,
        train_ids,
        val_ids,
        test_ids,
        global_percent,
    )

    n_images = len(assignment)

    for _ in range(max_swaps):
        best_swap = None
        best_swap_score = best_score

        for i in range(n_images):
            for j in range(i + 1, n_images):
                if assignment[i] == assignment[j]:
                    continue

                candidate = assignment.copy()
                candidate[i], candidate[j] = candidate[j], candidate[i]

                train_ids, val_ids, test_ids = make_ids_from_assignment(candidate)

                candidate_score, _, _, _ = score_split(
                    image_counts,
                    train_ids,
                    val_ids,
                    test_ids,
                    global_percent,
                )

                if candidate_score < best_swap_score:
                    best_swap_score = candidate_score
                    best_swap = (i, j, candidate)

        if best_swap is None:
            break

        _, _, assignment = best_swap
        best_score = best_swap_score

    return assignment, best_score


def print_distribution(name, image_counts, ids, global_percent):
    counts = image_counts[ids].sum(axis=0)
    counts, percent, total = distribution_from_counts(counts)

    print(f"\n[class distribution] {name}")
    print(f"  images: {len(ids)}")
    print(f"  total pixels: {total:,}")

    for c in range(len(counts)):
        diff = percent[c] - global_percent[c]
        print(
            f"  {c} ({CLASS_NAMES[c]:>5s}): "
            f"{counts[c]:>12,d} pixels  "
            f"{percent[c]:6.2f}%  "
            f"diff from global: {diff:+6.2f} pp"
        )

    return {
        "ids": ids,
        "counts": counts.astype(int).tolist(),
        "percent": percent.tolist(),
        "total_pixels": int(total),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n===== BALANCED FINAL SPLIT SEARCH =====")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Images : {N_IMAGES}")
    print(f"Val    : {N_VAL}")
    print(f"Test   : {N_TEST}")
    print(f"Seed   : {SEED}")
    print("======================================\n")

    image_counts = load_label_counts(
        base=BASE,
        n_images=N_IMAGES,
        nclasses=NCLASSES,
    )

    global_counts = image_counts.sum(axis=0)
    _, global_percent, global_total = distribution_from_counts(global_counts)

    print("[global class distribution]")
    print(f"  total pixels: {global_total:,}")
    for c in range(NCLASSES):
        print(
            f"  {c} ({CLASS_NAMES[c]:>5s}): "
            f"{global_counts[c]:>12,d} pixels  {global_percent[c]:6.2f}%"
        )

    rng = np.random.default_rng(SEED)

    best_assignment = None
    best_score = float("inf")

    for r in range(N_RANDOM_RESTARTS):
        assignment = random_assignment(
            rng=rng,
            n_images=N_IMAGES,
            n_val=N_VAL,
            n_test=N_TEST,
        )

        assignment, score = greedy_improve_assignment(
            image_counts=image_counts,
            assignment=assignment,
            global_percent=global_percent,
            max_swaps=MAX_GREEDY_SWAPS,
        )

        if score < best_score:
            best_score = score
            best_assignment = assignment.copy()

    train_ids, val_ids, test_ids = make_ids_from_assignment(best_assignment)

    total_score, train_score, val_score, test_score = score_split(
        image_counts=image_counts,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        global_percent=global_percent,
    )

    print("\n===== BEST SPLIT FOUND =====")
    print(f"Total score : {total_score:.4f}")
    print(f"Train score : {train_score:.4f}")
    print(f"Val score   : {val_score:.4f}")
    print(f"Test score  : {test_score:.4f}")

    print(f"\nTRAIN_IDS = {train_ids}")
    print(f"VAL_IDS   = {val_ids}")
    print(f"TEST_IDS  = {test_ids}")

    train_dist = print_distribution(
        name="TRAIN",
        image_counts=image_counts,
        ids=train_ids,
        global_percent=global_percent,
    )

    val_dist = print_distribution(
        name="VAL / CALIBRATION",
        image_counts=image_counts,
        ids=val_ids,
        global_percent=global_percent,
    )

    test_dist = print_distribution(
        name="TEST",
        image_counts=image_counts,
        ids=test_ids,
        global_percent=global_percent,
    )

    result = {
        "dataset": DATASET_NAME,
        "n_images": N_IMAGES,
        "n_classes": NCLASSES,
        "class_names": CLASS_NAMES,
        "seed": SEED,
        "n_val": N_VAL,
        "n_test": N_TEST,
        "score": {
            "total": total_score,
            "train": train_score,
            "val": val_score,
            "test": test_score,
        },
        "global_distribution": {
            "counts": global_counts.astype(int).tolist(),
            "percent": global_percent.tolist(),
            "total_pixels": int(global_total),
        },
        "train": train_dist,
        "val_calibration": val_dist,
        "test": test_dist,
        "note": (
            "Split selected to make train, validation/calibration, and test "
            "class distributions close to the global dataset distribution."
        ),
    }

    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=4)

    print(f"\n[save] Balanced split saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()