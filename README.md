# HYPSO Onboard Segmentation

Code for sea, land, and cloud segmentation of HYPSO-2 hyperspectral satellite imagery. The repository includes dataset preprocessing, FP32 model training, quantization-aware training, and ZCU104/Vitis AI DPU inference scripts.

## Repository structure

```text
dataset_processing/      Prepare HYPSO datasets and labels
model_definition/        Model architectures used in the thesis
train_fp32/              FP32 training scripts for comparison and deployment
quantization/            INT8 quantization-aware training scripts
zcu104_dpu_inference/    Vitis AI DPU inference on ZCU104
```

## Thesis context

This repository contains the code developed for the master's thesis "Deep Learning-Based Sea-Land-Cloud Segmentation of HYPSO Hyperspectral Imagery for FPGA Deployment with Vitis AI"
