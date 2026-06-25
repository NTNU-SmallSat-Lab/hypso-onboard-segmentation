from pathlib import Path
import re
import csv
import time
import gc
import numpy as np
import torch

from model_loader import MODEL_REGISTRY, build_model



try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False


dataset_name = "dataset_name"
base = Path("/home/datasets") / dataset_name

n_classes = 3

# Benchmark settings
BENCHMARK_PIXEL_BATCH = 65536
BENCHMARK_IMAGE_MAX_SIDE = 256
WARMUP = 20
REPEATS = 100

out_csv = Path("model_complexity_summary.csv")
out_tex = Path("model_complexity_latex_rows.txt")


SKIP_MODELS = {
    
}


# Thesis display names.
DISPLAY_NAMES = {
    "spectralpointwise": "1D-SpectralPointwise",
    "sp_unet_large": "2D-SpecPW-UNet-Large",
    "sp_unet_medium": "2D-SpecPW-UNet-Medium",
    "sp_unet_small": "2D-SpecPW-UNet-Small",
    "justoliunet": "1D-Justo-LiuNet",
    "justoliunet_bn": "1D-Justo-LiuNet-BN",
    "justounetsimple": "2D-Justo-UNet-Simple",
    "waveletcnn": "2D-WaveletCNN",
}


def first_id(base: Path) -> int:
    files = sorted(base.glob("data*.npy"))

    if not files:
        raise FileNotFoundError(f"No data*.npy files found in {base}")

    ids = []

    for file in files:
        m = re.search(r"data(\d+)\.npy$", file.name)
        if m is not None:
            ids.append(int(m.group(1)))

    if not ids:
        raise FileNotFoundError(f"No files matching data<number>.npy found in {base}")

    return min(ids)


def display_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


def model_type(name: str) -> str:
    n = name.lower()

    if "wavelet" in n:
        return "2D wavelet"

    if "1d" in n or "liunet" in n or "pointwise" in n:
        return "1D spectral"

    if "unet" in n or "cnn" in n or "cfcn" in n or "cunet" in n:
        return "2D spectral-spatial"

    return "Unknown"


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def parameter_size_mb(model):
    size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return size_bytes / (1024 ** 2)


def shape_str(x):
    return "x".join(str(v) for v in tuple(x.shape))


def make_inputs(x, in_channels, device):
    H, W, _ = x.shape

    h_bench = min(H, BENCHMARK_IMAGE_MAX_SIDE)
    w_bench = min(W, BENCHMARK_IMAGE_MAX_SIDE)

    pixel_batch = min(H * W, BENCHMARK_PIXEL_BATCH)

    candidates = [
        (
            "pixel",
            torch.randn(
                pixel_batch,
                in_channels,
                dtype=torch.float32,
                device=device,
            ),
        ),
        (
            "image_nchw",
            torch.randn(
                1,
                in_channels,
                h_bench,
                w_bench,
                dtype=torch.float32,
                device=device,
            ),
        ),
        (
            "image_nhwc",
            torch.randn(
                1,
                h_bench,
                w_bench,
                in_channels,
                dtype=torch.float32,
                device=device,
            ),
        ),
    ]

    return candidates


def find_valid_input(model, candidates):
    model.eval()

    with torch.no_grad():
        for input_mode, dummy in candidates:
            try:
                out = model(dummy)
                return input_mode, dummy, out
            except Exception:
                continue

    return None, None, None


def benchmark_inference(model, dummy, device):
    model.eval()

    with torch.no_grad():
        # Warmup iterations are excluded from timing.
        for _ in range(WARMUP):
            _ = model(dummy)

        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            times = []

            for _ in range(REPEATS):
                start_event.record()
                _ = model(dummy)
                end_event.record()

                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event))

            peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        else:
            times = []
            peak_mem_mb = np.nan

            for _ in range(REPEATS):
                t0 = time.perf_counter()
                _ = model(dummy)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

    return float(np.mean(times)), float(np.std(times)), float(peak_mem_mb)


def compute_macs(model, dummy):
    if not HAS_THOP:
        return np.nan, np.nan

    try:
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        flops = 2 * macs
        return macs / 1e6, flops / 1e6
    except Exception:
        return np.nan, np.nan


def fmt_num(x, decimals=2):
    if isinstance(x, float) and np.isnan(x):
        return "N/A"

    return f"{x:.{decimals}f}"


i = first_id(base)

x = np.load(base / f"data{i}.npy", mmap_mode="r")
y = np.load(base / f"label{i}.npy", mmap_mode="r")

H, W, in_channels = x.shape
n_pixels = H * W

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Dataset: {base}")
print(f"Using: data{i}.npy / label{i}.npy")
print(f"data shape: {x.shape}")
print(f"label shape: {y.shape}")
print(f"in_channels: {in_channels}")
print(f"n_classes: {n_classes}")
print(f"pixels per image: {n_pixels:,}")
print(f"device: {device}")
print(f"THOP available for MACs/FLOPs: {HAS_THOP}\n")


header = [
    "Model",
    "Type",
    "Parameters",
    "Trainable parameters",
    "Parameter size MB",
    "Input mode",
    "Benchmark input shape",
    "Output shape",
    "MACs M",
    "FLOPs M",
    "Inference ms",
    "Inference SD ms",
    "Peak memory MB",
]

rows = []

print("-" * 150)
print(
    f"{'Model':28s} "
    f"{'Type':22s} "
    f"{'Params':>10s} "
    f"{'Param MB':>9s} "
    f"{'Input mode':>12s} "
    f"{'Input shape':>22s} "
    f"{'Output shape':>22s} "
    f"{'MACs M':>10s} "
    f"{'Inf ms':>10s} "
    f"{'Peak MB':>10s}"
)
print("-" * 150)


for name_model in MODEL_REGISTRY:
    if name_model in SKIP_MODELS:
        continue

    model = None
    dummy = None

    try:
        model = build_model(name_model, in_channels, n_classes).to(device)
        model.eval()

        total_params, trainable_params = count_parameters(model)
        param_size_mb = parameter_size_mb(model)

        candidates = make_inputs(x, in_channels, device)
        input_mode, dummy, out = find_valid_input(model, candidates)

        if dummy is None:
            row = [
                display_name(name_model),
                model_type(name_model),
                total_params,
                trainable_params,
                param_size_mb,
                "FAILED",
                "FAILED",
                "FAILED",
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ]

            rows.append(row)

            print(
                f"{display_name(name_model):28s} "
                f"{model_type(name_model):22s} "
                f"{total_params:10,d} "
                f"{param_size_mb:9.3f} "
                f"{'FAILED':>12s}"
            )

            del candidates

            continue

        input_shape = shape_str(dummy)
        output_shape = shape_str(out)

        del out
        del candidates

        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        macs_m, flops_m = compute_macs(model, dummy)

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        inf_ms, inf_sd_ms, peak_mem_mb = benchmark_inference(model, dummy, device)

        row = [
            display_name(name_model),
            model_type(name_model),
            total_params,
            trainable_params,
            param_size_mb,
            input_mode,
            input_shape,
            output_shape,
            macs_m,
            flops_m,
            inf_ms,
            inf_sd_ms,
            peak_mem_mb,
        ]

        rows.append(row)

        print(
            f"{display_name(name_model):28s} "
            f"{model_type(name_model):22s} "
            f"{total_params:10,d} "
            f"{param_size_mb:9.3f} "
            f"{input_mode:>12s} "
            f"{input_shape:>22s} "
            f"{output_shape:>22s} "
            f"{fmt_num(macs_m):>10s} "
            f"{fmt_num(inf_ms):>10s} "
            f"{fmt_num(peak_mem_mb):>10s}"
        )

    except Exception as e:
        print(f"{display_name(name_model):28s} FAILED: {e}")

    finally:
        if model is not None:
            del model

        if dummy is not None:
            del dummy

        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


print("-" * 150)


with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)


with open(out_tex, "w") as f:
    f.write("% LaTeX table rows\n")
    f.write(
        "% Model & Parameters & Parameter size MB & MACs M "
        "& Inference ms & Peak memory MB \\\\\n"
    )

    for row in rows:
        (
            model_name,
            typ,
            total_params,
            trainable_params,
            param_size_mb,
            input_mode,
            input_shape,
            output_shape,
            macs_m,
            flops_m,
            inf_ms,
            inf_sd_ms,
            peak_mem_mb,
        ) = row

        f.write(
            f"{model_name} & "
            f"{total_params:,} & "
            f"{param_size_mb:.3f} & "
            f"{fmt_num(macs_m)} & "
            f"{fmt_num(inf_ms)} & "
            f"{fmt_num(peak_mem_mb)} \\\\\n"
        )


print(f"\nSaved CSV summary to: {out_csv.resolve()}")
print(f"Saved LaTeX rows to: {out_tex.resolve()}")

print("\nSuggested thesis columns:")
print(
    "Model | Parameters | Parameter size (MB) | MACs (M) | "
    "Inference time (ms) | Peak memory (MB)"
)