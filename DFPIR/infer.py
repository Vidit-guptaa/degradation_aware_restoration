"""
DFPIR – Inference Script
Run on a single image or a folder of images.

Usage examples
──────────────
# Auto-detect degradation and restore:
python infer.py --input noisy_image.png --task denoising --checkpoint dfpir_final.pth

# Restore a whole folder:
python infer.py --input ./hazy_images/ --task dehazing --checkpoint dfpir_final.pth --output ./restored/

# Available tasks: denoising | deraining | dehazing | deblurring | lowlight
"""

import os, argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms.functional as TF

from model import DFPIR

# ─────────────────────────────────────────────────────────────────────────────
TASK_IDS = {
    "denoising":  0,
    "deraining":  1,
    "dehazing":   2,
    "deblurring": 3,
    "lowlight":   4,
}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint: str, device, dim=48, prompt_dim=512) -> DFPIR:
    model = DFPIR(dim=dim, prompt_dim=prompt_dim, num_tasks=5).to(device)
    if checkpoint and os.path.exists(checkpoint):
        state = torch.load(checkpoint, map_location=device)
        # Handle wrapped state-dicts
        if "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        print(f"✓ Loaded weights from {checkpoint}")
    else:
        print("⚠  No checkpoint found – running with random weights (demo only).")
    model.eval()
    return model


@torch.no_grad()
def restore_image(model, img: Image.Image, task_id: int,
                  device, tile_size: int = 512, overlap: int = 32) -> Image.Image:
    """
    Restore a single PIL image.
    Tiles large images automatically to avoid OOM.
    """
    W, H = img.size

    # Convert to tensor  [0, 1]
    x = TF.to_tensor(img).unsqueeze(0).to(device)          # (1, 3, H, W)
    tid = torch.tensor([task_id], dtype=torch.long, device=device)

    # Pad to multiples of 8 so up/down-sampling stays aligned
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    _, _, pH, pW = x.shape

    if pH <= tile_size and pW <= tile_size:
        # Fit in memory – process whole image
        out = model(x, task_id=tid)
    else:
        # Tile-based inference
        out = torch.zeros_like(x)
        cnt = torch.zeros_like(x)
        step = tile_size - overlap
        for row in range(0, pH, step):
            for col in range(0, pW, step):
                r1, r2 = row, min(row + tile_size, pH)
                c1, c2 = col, min(col + tile_size, pW)
                tile    = x[:, :, r1:r2, c1:c2]
                tile_out = model(tile, task_id=tid)
                out[:, :, r1:r2, c1:c2] += tile_out
                cnt[:, :, r1:r2, c1:c2] += 1
        out = out / cnt.clamp(min=1)

    # Remove padding
    out = out[:, :, :H, :W]
    out = out.squeeze(0).clamp(0, 1).cpu()
    return TF.to_pil_image(out)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser("DFPIR Inference")
    p.add_argument("--input",      required=True,
                   help="Path to input image OR directory of images")
    p.add_argument("--task",       required=True,
                   choices=list(TASK_IDS.keys()),
                   help="Degradation type to restore")
    p.add_argument("--checkpoint", default="checkpoints/dfpir_final.pth",
                   help="Path to trained model weights (.pth)")
    p.add_argument("--output",     default="restored",
                   help="Output directory (default: ./restored/)")
    p.add_argument("--dim",        type=int, default=48,
                   help="Base channel dim (must match training)")
    p.add_argument("--tile",       type=int, default=512,
                   help="Tile size for large images")
    p.add_argument("--overlap",    type=int, default=32,
                   help="Tile overlap to reduce seam artefacts")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    model   = load_model(args.checkpoint, device, dim=args.dim)
    task_id = TASK_IDS[args.task]
    os.makedirs(args.output, exist_ok=True)

    # Collect input images
    inp = Path(args.input)
    if inp.is_file():
        paths = [inp]
    else:
        paths = [p for p in inp.iterdir() if p.suffix.lower() in IMG_EXTS]

    print(f"Restoring {len(paths)} image(s) | task: {args.task}")

    for path in tqdm(paths):
        img  = Image.open(path).convert("RGB")
        rest = restore_image(model, img, task_id, device,
                             tile_size=args.tile, overlap=args.overlap)
        out_path = os.path.join(args.output, path.name)
        rest.save(out_path)

    print(f"\n✓ Done. Results saved to: {args.output}/")


if __name__ == "__main__":
    main()
