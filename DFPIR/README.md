# DFPIR – Degradation-Aware Feature Perturbation for All-in-One Image Restoration
### CVPR 2025 Implementation

Feed in a noisy / hazy / rainy / blurry / dark image → get a clean image back.

---

## What this code does

This implements the full **DFPIR** pipeline from the paper:

| Component | What it does |
|-----------|-------------|
| `TransformerBlock` | Restormer backbone blocks (encoder + decoder) |
| `DGCPM` | Expands channels 2×, reorders them by degradation type (channel-shuffle) |
| `CAAPM` | Cross-attention between shuffled & original features; masks attention maps with top-K |
| `DGPB` | Combines DGCPM + CAAPM; inserted at every skip connection |
| `DFPIR` | Full 4-level encoder-decoder with 4 DGPBs |

**Supported degradations:** Denoising · Deraining · Dehazing · Deblurring · Low-light

---

## Requirements – Software & Hardware

| Item | Minimum | Recommended |
|------|---------|-------------|
| **OS** | Windows 10 / Ubuntu 20.04 / macOS 12 | Ubuntu 22.04 |
| **Python** | 3.9 | 3.10 or 3.11 |
| **PyTorch** | 2.0 (CPU works) | 2.2 + CUDA 11.8 |
| **GPU** | None (slow) | NVIDIA RTX 3090 24 GB |
| **RAM** | 8 GB | 16 GB |

---

## Step-by-Step Setup

### Step 1 – Install Python
Download Python 3.10 from https://www.python.org/downloads/  
*(Tick "Add Python to PATH" during install on Windows)*

### Step 2 – Create a virtual environment
```bash
# Windows
python -m venv dfpir_env
dfpir_env\Scripts\activate

# macOS / Linux
python3 -m venv dfpir_env
source dfpir_env/bin/activate
```

### Step 3 – Install PyTorch
Go to https://pytorch.org/get-started/locally/ and copy the right command.

**CUDA (NVIDIA GPU) – recommended:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CPU only:**
```bash
pip install torch torchvision
```

### Step 4 – Install remaining dependencies
```bash
pip install -r requirements.txt
```

---

## Quick Test (no training needed)

```bash
python infer.py \
  --input  path/to/my_blurry_photo.jpg \
  --task   deblurring \
  --checkpoint checkpoints/dfpir_final.pth
```
The restored image will appear in `./restored/`.

If you don't have a checkpoint yet, the script still runs with random weights
(output won't be meaningful – you need to train first).

---

## Training

### Prepare data
Organise your pairs like this:
```
data/
  denoising/
    clean/      ← ground-truth images
    degraded/   ← noisy images (same filenames)
  deraining/
    clean/
    degraded/
  dehazing/
    clean/
    degraded/
  deblurring/
    clean/
    degraded/
  lowlight/
    clean/
    degraded/
```
You only need the tasks you want to train on. Missing folders are skipped.

**Recommended public datasets (download separately):**
| Task | Dataset | Link |
|------|---------|------|
| Denoising | BSD400 + WED | Search "BSD400 dataset" |
| Deraining | Rain100L | Search "Rain100L dataset" |
| Dehazing | RESIDE SOTS | Search "RESIDE dataset" |
| Deblurring | GoPro | Search "GoPro deblur dataset" |
| Low-light | LOL-v1 | Search "LOL low-light dataset" |

### Run training
```bash
python train.py \
  --data_root  data/ \
  --save_dir   checkpoints/ \
  --dim        48 \
  --batch_size 5 \
  --patch_size 128 \
  --epochs     80
```

### Training stages (matches paper exactly)
| Stage | Epochs | Patch size | Batch | LR |
|-------|--------|------------|-------|----|
| Main  | 1-80   | 128×128    | 5     | 1e-4 |
| Fine-tune | 81-85 | 192×192 | 3     | 1e-5 |

Training on RTX 3090 takes ~24–48 hours for 80 epochs on all 5 tasks.

---

## Inference on your own image

```bash
# Restore a single noisy image
python infer.py --input noisy.png --task denoising --checkpoint checkpoints/dfpir_final.pth

# Restore all images in a folder (hazy photos)
python infer.py --input ./hazy_folder/ --task dehazing --checkpoint checkpoints/dfpir_final.pth --output ./output/

# For very large images (>512px) tiling is automatic
python infer.py --input large_image.jpg --task deraining --checkpoint checkpoints/dfpir_final.pth --tile 512 --overlap 32
```

### Task options
| `--task` value | Use for |
|----------------|---------|
| `denoising`    | Images with Gaussian / real noise |
| `deraining`    | Rainy images with streaks |
| `dehazing`     | Foggy / hazy images |
| `deblurring`   | Motion-blurred images |
| `lowlight`     | Dark / underexposed images |

---

## File Overview

```
DFPIR/
├── model.py        ← Full network: DFPIR, DGPB, DGCPM, CAAPM, TransformerBlock
├── train.py        ← Training loop + dataset loader
├── infer.py        ← Run on your images
├── requirements.txt
└── README.md
```

---

## Common Issues

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: einops` | `pip install einops` |
| CUDA out of memory | Reduce `--batch_size` or use `--tile 256` at inference |
| `RuntimeError: size mismatch` | Make sure `--dim` matches what you trained with |
| Slow on CPU | Normal; GPU is 20-50× faster |

---

## Citation
```
@inproceedings{tian2025dfpir,
  title={Degradation-Aware Feature Perturbation for All-in-One Image Restoration},
  author={Tian, Xiangpeng and Liao, Xiangyu and Liu, Xiao and Li, Meng and Ren, Chao},
  booktitle={CVPR},
  year={2025}
}
```
Official code: https://github.com/TxpHome/DFPIR
