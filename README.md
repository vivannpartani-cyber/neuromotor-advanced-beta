# Neuromotor (Advanced Beta)

> **Production-grade Brain-to-Image Neural Decoding**
> A scaled architecture for training on the massive Natural Scenes Dataset (NSD) using multi-GPU clusters.

This repository is an advanced, cluster-ready evolution of the Neuromotor Proof-of-Concept. It upgrades the pipeline to support massive datasets, distributed training, and deep residual neural architectures.

---

## Key Features

1. **Multi-GPU Distributed Training**: Powered by HuggingFace `Accelerate`. Seamlessly scales from 1 GPU to a massive cluster (e.g., 8x NVIDIA H100s) with zero code changes.
2. **Deep Residual Architecture**: A custom `DeepResidualMapper` capable of handling the extreme variance of the 73,000 NSD scans without vanishing gradients.
3. **High-Performance Dataloader**: Custom PyTorch `NSDDataset` that safely streams HDF5 files to prevent RAM exhaustion when working with hundreds of gigabytes of data.
4. **Cloud Logging**: Native Weights & Biases (`wandb`) integration for monitoring training loss from your dashboard.

---

## Installation

Clone this repository onto your cloud GPU instance (e.g., RunPod, AWS EC2, or Lambda Labs):

```bash
git clone https://github.com/vivannpartani-cyber/neuromotor-advanced-beta.git
cd neuromotor-advanced-beta
pip install -r requirements.txt
```

Initialize your multi-GPU environment using Accelerate:
```bash
accelerate config
```
*(Simply answer the prompts regarding how many GPUs you have and if you want to use FP16/BF16 mixed precision).*

---

## Dataset Preparation (NSD)

You must explicitly sign a data-use agreement to access the Natural Scenes Dataset.
1. Apply for access at [naturalscenesdataset.org](http://naturalscenesdataset.org/).
2. Download the fMRI betas and the pre-computed COCO CLIP embeddings for your target subject.
3. Convert/Format them into PyTorch-friendly HDF5 files (`fmri.h5` and `clip.h5`).

---

## Training on the Cluster

Once your data is ready, launch the distributed training run:

```bash
accelerate launch train.py \
    --fmri_path /path/to/nsd/subj01_fmri.h5 \
    --clip_path /path/to/nsd/subj01_clip.h5 \
    --subject subj01 \
    --input_dim 10000 \
    --batch_size 256 \
    --epochs 200 \
    --wandb_project "neuromotor-nsd-run1"
```

### What happens?
- `accelerate` will spawn a process for every GPU on your machine.
- The `NSDDataset` will lazily stream data from the fast NVMe storage into VRAM.
- `wandb` will sync your metrics to the cloud.
- Checkpoints will be saved in `./checkpoints/` every 10 epochs.

---
*Built for production scale by Vivann Partani.*
