#!/usr/bin/env python3
"""
 Python VART runner for tiled hyperspectral segmentation on ZCU104.
 
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import xir
import vart


DEFAULT_TEST_IDS = [12, 16, 17, 23, 24, 46, 54, 58]


def parse_ids(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def vart_dtype_to_numpy(dtype) -> np.dtype:
    s = str(dtype).lower()
    if "int8" in s and "uint8" not in s:
        return np.dtype(np.int8)
    if "uint8" in s:
        return np.dtype(np.uint8)
    if "int16" in s and "uint16" not in s:
        return np.dtype(np.int16)
    if "uint16" in s:
        return np.dtype(np.uint16)
    if "int32" in s and "uint32" not in s:
        return np.dtype(np.int32)
    if "uint32" in s:
        return np.dtype(np.uint32)
    if "float32" in s or "float" in s:
        return np.dtype(np.float32)
    raise RuntimeError(f"Unsupported VART tensor dtype: {dtype!r} / {s}")


def get_dpu_subgraph(graph: "xir.Graph"):
    root = graph.get_root_subgraph()
    children = root.toposort_child_subgraph()
    dpu_subgraphs = [
        sg for sg in children
        if sg.has_attr("device") and str(sg.get_attr("device")).upper() == "DPU"
    ]
    if not dpu_subgraphs:
        raise RuntimeError("No DPU subgraph found. Use the compiled ZCU104 xmodel.")
    if len(dpu_subgraphs) > 1:
        print(f"[warn] Found {len(dpu_subgraphs)} DPU subgraphs; using the first.")
    return dpu_subgraphs[0]


def tensor_fix_scale(tensor):
    if tensor.has_attr("fix_point"):
        fix = int(tensor.get_attr("fix_point"))
        return fix, float(2 ** fix)
    return None, 1.0


def infer_input_layout(shape: tuple[int, ...], channels: int):
    if len(shape) != 4 or shape[0] != 1:
        raise RuntimeError(f"Expected batch-1 4D input tensor, got {shape}")
    if shape[1] == channels:
        return "NCHW", int(shape[2]), int(shape[3])
    if shape[-1] == channels:
        return "NHWC", int(shape[1]), int(shape[2])
    raise RuntimeError(f"Cannot infer input layout from shape {shape} and channels={channels}")


def infer_output_layout(shape: tuple[int, ...], nclasses: int):
    if len(shape) != 4 or shape[0] != 1:
        raise RuntimeError(f"Expected batch-1 4D output tensor, got {shape}")
    if shape[1] == nclasses:
        return "NCHW"
    if shape[-1] == nclasses:
        return "NHWC"
    raise RuntimeError(f"Cannot infer output layout from shape {shape} and nclasses={nclasses}")


def make_starts(length: int, patch: int, edge_cover: bool) -> list[int]:
    if length < patch:
        raise RuntimeError(f"Image dimension {length} is smaller than patch={patch}")
    starts = list(range(0, length - patch + 1, patch))
    if edge_cover:
        last = length - patch
        if starts[-1] != last:
            starts.append(last)
    return starts


def update_confusion_matrix(cm: np.ndarray, pred: np.ndarray, gt: np.ndarray, nclasses: int, ignore_index: int | None):
    valid = (gt >= 0) & (gt < nclasses)
    if ignore_index is not None:
        valid &= gt != ignore_index

    if not np.any(valid):
        return

    p = pred[valid].astype(np.int64)
    g = gt[valid].astype(np.int64)

    cm += np.bincount(
        nclasses * g + p,
        minlength=nclasses * nclasses,
    ).reshape(nclasses, nclasses)


def metrics_from_cm(cm: np.ndarray):
    tp = np.diag(cm).astype(np.float64)
    den = cm.sum(axis=1) + cm.sum(axis=0) - tp

    iou = np.full(cm.shape[0], np.nan, dtype=np.float64)
    present = den > 0
    iou[present] = tp[present] / den[present]

    total = cm.sum()
    acc = float(tp.sum() / total) if total > 0 else 0.0
    miou = float(np.nanmean(iou)) if np.any(present) else 0.0
    return miou, iou, acc


def quantize_full_image_hwc(x: np.ndarray, input_scale: float, input_dtype: np.dtype):
    """Quantize an HWC float image once. Returns HWC quantized image."""
    x = np.asarray(x, dtype=np.float32)

    if input_dtype == np.dtype(np.int8):
        return np.clip(np.round(x * input_scale), -128, 127).astype(np.int8)
    if input_dtype == np.dtype(np.uint8):
        return np.clip(np.round(x * input_scale), 0, 255).astype(np.uint8)
    if input_dtype == np.dtype(np.float32):
        return x.astype(np.float32, copy=False)

    raise RuntimeError(f"Unsupported input tensor dtype after mapping: {input_dtype}")


def output_to_pred(out: np.ndarray, output_layout: str):
    if output_layout == "NCHW":
        return out[0].argmax(axis=0).astype(np.uint8)
    if output_layout == "NHWC":
        return out[0].argmax(axis=-1).astype(np.uint8)
    raise RuntimeError(f"Unknown output layout: {output_layout}")


def run_one_image_fast(
    runner,
    input_tensor,
    output_tensor,
    input_scale: float,
    data_path: Path,
    label_path: Path | None,
    patch: int,
    nclasses: int,
    ignore_index: int | None,
    edge_cover: bool,
    out_dir: Path,
    image_id: int,
    save_preds: bool,
    save_logits: bool,
):
    t_load0 = time.time()
    x = np.load(data_path)  # load into RAM once
    if x.ndim != 3:
        raise RuntimeError(f"Expected HWC data in {data_path}, got {x.shape}")
    h, w, c = x.shape
    t_load = time.time() - t_load0

    input_shape = tuple(input_tensor.dims)
    output_shape = tuple(output_tensor.dims)

    input_layout, pin_h, pin_w = infer_input_layout(input_shape, c)
    output_layout = infer_output_layout(output_shape, nclasses)

    if pin_h != patch or pin_w != patch:
        raise RuntimeError(
            f"xmodel expects {pin_h}x{pin_w}, but --patch is {patch}. "
            "Use the patch size used during export/compile."
        )

    input_dtype = vart_dtype_to_numpy(input_tensor.dtype)
    output_dtype = vart_dtype_to_numpy(output_tensor.dtype)

    t_quant0 = time.time()
    xq_hwc = quantize_full_image_hwc(x, input_scale, input_dtype)
    del x
    if input_layout == "NCHW":
        # Pretranspose once instead of transposing each tile.
        xq = np.transpose(xq_hwc, (2, 0, 1))
    else:
        xq = xq_hwc
    t_quant = time.time() - t_quant0

    inp = np.empty(input_shape, dtype=input_dtype)
    out = np.empty(output_shape, dtype=output_dtype)

    y_starts = make_starts(h, patch, edge_cover=edge_cover)
    x_starts = make_starts(w, patch, edge_cover=edge_cover)

    pred_full = np.full((h, w), 255, dtype=np.uint8)
    filled = np.zeros((h, w), dtype=bool)

    logits_full = None
    if save_logits:
        logits_full = np.zeros((nclasses, h, w), dtype=np.int16)

    dpu_seconds = 0.0
    tile_copy_seconds = 0.0
    post_seconds = 0.0
    tiles = 0

    for y0 in y_starts:
        for x0 in x_starts:
            t_copy0 = time.time()
            if input_layout == "NCHW":
                inp[0, :, :, :] = xq[:, y0:y0 + patch, x0:x0 + patch]
            else:
                inp[0, :, :, :] = xq[y0:y0 + patch, x0:x0 + patch, :]
            tile_copy_seconds += time.time() - t_copy0

            t_dpu0 = time.time()
            job_id = runner.execute_async([inp], [out])
            runner.wait(job_id)
            dpu_seconds += time.time() - t_dpu0

            t_post0 = time.time()
            pred_patch = output_to_pred(out, output_layout)

            area = pred_full[y0:y0 + patch, x0:x0 + patch]
            fill_mask = ~filled[y0:y0 + patch, x0:x0 + patch]
            area[fill_mask] = pred_patch[fill_mask]
            filled[y0:y0 + patch, x0:x0 + patch][fill_mask] = True

            if save_logits and logits_full is not None:
                if output_layout == "NCHW":
                    out_chw = out[0].astype(np.int16)
                else:
                    out_chw = np.transpose(out[0], (2, 0, 1)).astype(np.int16)
                for cls in range(nclasses):
                    target = logits_full[cls, y0:y0 + patch, x0:x0 + patch]
                    target[fill_mask] = out_chw[cls][fill_mask]

            post_seconds += time.time() - t_post0
            tiles += 1

    if not filled.all():
        print(f"[warn] image {image_id}: {(~filled).sum()} pixels not covered")

    save_seconds = 0.0
    pred_path = None
    if save_preds:
        t_save0 = time.time()
        pred_path = out_dir / f"pred{image_id}.npy"
        np.save(pred_path, pred_full)
        if save_logits and logits_full is not None:
            np.save(out_dir / f"logits_int8_{image_id}.npy", logits_full)
        save_seconds = time.time() - t_save0

    result = {
        "image_id": image_id,
        "height": h,
        "width": w,
        "channels": c,
        "tiles": tiles,
        "load_seconds": t_load,
        "quant_seconds": t_quant,
        "copy_seconds": tile_copy_seconds,
        "dpu_seconds": dpu_seconds,
        "post_seconds": post_seconds,
        "save_seconds": save_seconds,
        "pred_path": None if pred_path is None else str(pred_path),
    }

    if label_path is not None and label_path.exists():
        y = np.load(label_path, mmap_mode="r")
        if y.shape != (h, w):
            raise RuntimeError(f"Label shape mismatch for image {image_id}: label={y.shape}, data={(h, w)}")
        cm = np.zeros((nclasses, nclasses), dtype=np.int64)
        update_confusion_matrix(cm, pred_full, y, nclasses, ignore_index)
        miou, iou, acc = metrics_from_cm(cm)
        result.update({"miou": miou, "iou": iou, "acc": acc, "cm": cm})

    return result


def main():
    parser = argparse.ArgumentParser(description="Faster Python ZCU104 VART hyperspectral segmentation runner")
    parser.add_argument("--xmodel", type=str, required=True, help="Compiled ZCU104 DPU .xmodel")
    parser.add_argument("--dataset_root", type=str, default="dataset")
    parser.add_argument("--test_ids", type=parse_ids, default=DEFAULT_TEST_IDS)
    parser.add_argument("--patch", type=int, default=64)
    parser.add_argument("--nclasses", type=int, default=3)
    parser.add_argument("--ignore_index", type=int, default=-100)
    parser.add_argument("--out_dir", type=str, default="dpu_predictions_fast")
    parser.add_argument("--n_warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--no_edge_cover", action="store_true")
    parser.add_argument("--save_preds", action="store_true", help="Save pred<ID>.npy files. Disabled by default for faster benchmarking.")
    parser.add_argument("--save_logits", action="store_true")
    args = parser.parse_args()

    xmodel_path = Path(args.xmodel)
    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not xmodel_path.exists():
        raise FileNotFoundError(f"xmodel not found: {xmodel_path}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")

    print(f"[info] xmodel       : {xmodel_path}")
    print(f"[info] dataset_root : {dataset_root}")
    print(f"[info] test_ids     : {args.test_ids}")
    print(f"[info] patch        : {args.patch}")
    print(f"[info] out_dir      : {out_dir}")
    print(f"[info] save_preds   : {args.save_preds}")

    graph = xir.Graph.deserialize(str(xmodel_path))
    dpu_sg = get_dpu_subgraph(graph)
    runner = vart.Runner.create_runner(dpu_sg, "run")

    input_tensor = runner.get_input_tensors()[0]
    output_tensor = runner.get_output_tensors()[0]

    input_fix, input_scale = tensor_fix_scale(input_tensor)
    output_fix, output_scale = tensor_fix_scale(output_tensor)

    print(
        f"[runner] input  shape={tuple(input_tensor.dims)}, dtype={input_tensor.dtype}, "
        f"np_dtype={vart_dtype_to_numpy(input_tensor.dtype)}, fix_point={input_fix}, scale={input_scale}"
    )
    print(
        f"[runner] output shape={tuple(output_tensor.dims)}, dtype={output_tensor.dtype}, "
        f"np_dtype={vart_dtype_to_numpy(output_tensor.dtype)}, fix_point={output_fix}, scale={output_scale}"
    )

    edge_cover = not args.no_edge_cover

    first_id = args.test_ids[0]
    for i in range(args.n_warmup):
        print(f"[warmup] {i + 1}/{args.n_warmup} on image {first_id}")
        run_one_image_fast(
            runner=runner,
            input_tensor=input_tensor,
            output_tensor=output_tensor,
            input_scale=input_scale,
            data_path=dataset_root / f"data{first_id}.npy",
            label_path=dataset_root / f"label{first_id}.npy",
            patch=args.patch,
            nclasses=args.nclasses,
            ignore_index=args.ignore_index,
            edge_cover=edge_cover,
            out_dir=out_dir,
            image_id=first_id,
            save_preds=False,
            save_logits=False,
        )

    all_results = []
    cm_total = np.zeros((args.nclasses, args.nclasses), dtype=np.int64)

    for rep in range(args.repeat):
        print(f"\n[run] repeat {rep + 1}/{args.repeat}")

        for image_id in args.test_ids:
            data_path = dataset_root / f"data{image_id}.npy"
            label_path = dataset_root / f"label{image_id}.npy"

            if not data_path.exists():
                raise FileNotFoundError(f"Missing data file: {data_path}")

            wall0 = time.time()
            result = run_one_image_fast(
                runner=runner,
                input_tensor=input_tensor,
                output_tensor=output_tensor,
                input_scale=input_scale,
                data_path=data_path,
                label_path=label_path if label_path.exists() else None,
                patch=args.patch,
                nclasses=args.nclasses,
                ignore_index=args.ignore_index,
                edge_cover=edge_cover,
                out_dir=out_dir,
                image_id=image_id,
                save_preds=args.save_preds,
                save_logits=args.save_logits,
            )
            result["wall_seconds"] = time.time() - wall0
            result["repeat"] = rep
            all_results.append(result)

            if "cm" in result:
                cm_total += result["cm"]

            host_non_dpu = (
                result["load_seconds"]
                + result["quant_seconds"]
                + result["copy_seconds"]
                + result["post_seconds"]
                + result["save_seconds"]
            )

            msg = (
                f"[image {image_id}] H={result['height']} W={result['width']} C={result['channels']} "
                f"tiles={result['tiles']} "
                f"dpu={result['dpu_seconds'] * 1000:.2f} ms "
                f"wall={result['wall_seconds'] * 1000:.2f} ms "
                f"host={host_non_dpu * 1000:.2f} ms "
                f"(load={result['load_seconds'] * 1000:.1f}, quant={result['quant_seconds'] * 1000:.1f}, "
                f"copy={result['copy_seconds'] * 1000:.1f}, post={result['post_seconds'] * 1000:.1f}, "
                f"save={result['save_seconds'] * 1000:.1f})"
            )

            if "miou" in result:
                msg += f" acc={result['acc']:.6f} mIoU={result['miou']:.6f} IoU={np.round(result['iou'], 6).tolist()}"

            print(msg)

    total_dpu = sum(r["dpu_seconds"] for r in all_results)
    total_wall = sum(r["wall_seconds"] for r in all_results)
    total_images = len(all_results)

    print("\n[summary]")
    print(f"images evaluated : {total_images}")
    print(f"total DPU time   : {total_dpu:.6f} s")
    print(f"total wall time  : {total_wall:.6f} s")
    print(f"avg DPU latency  : {(total_dpu / max(total_images, 1)) * 1000:.2f} ms/image")
    print(f"avg wall latency : {(total_wall / max(total_images, 1)) * 1000:.2f} ms/image")
    print(f"avg DPU FPS      : {total_images / max(total_dpu, 1e-12):.2f}")
    print(f"avg wall FPS     : {total_images / max(total_wall, 1e-12):.2f}")

    if cm_total.sum() > 0:
        miou, iou, acc = metrics_from_cm(cm_total)
        print(f"aggregate acc    : {acc:.6f}")
        print(f"aggregate mIoU   : {miou:.6f}")
        print(f"aggregate IoU    : {np.round(iou, 6).tolist()}")


if __name__ == "__main__":
    main()
