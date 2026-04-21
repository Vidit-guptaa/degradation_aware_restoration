"""
DFPIR – Training Script
Trains the model on multiple degradation types simultaneously (all-in-one).
"""

import os, random, argparse, math
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from model import DFPIR

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

TASK_IDS = {
    "denoising":   0,
    "deraining":   1,
    "dehazing":    2,
    "deblurring":  3,
    "lowlight":    4,
}

class DegradedDataset(Dataset):
    """
    Expects pairs of folders:
        data_root/
            denoising/clean/   + denoising/degraded/
            deraining/clean/   + deraining/degraded/
            dehazing/clean/    + dehazing/degraded/
            ...
    Each sub-folder must contain matching filenames.
    """
    def __init__(self, root, patch_size=128, augment=True):
        self.samples = []
        self.patch_size = patch_size
        self.augment = augment
        for task, tid in TASK_IDS.items():
            clean_dir = os.path.join(root, task, "clean")
            degr_dir  = os.path.join(root, task, "degraded")
            if not os.path.isdir(clean_dir):
                continue
            for fname in sorted(os.listdir(clean_dir)):
                cp = os.path.join(clean_dir, fname)
                dp = os.path.join(degr_dir,  fname)
                if os.path.exists(dp):
                    self.samples.append((dp, cp, tid))
        print(f"[Dataset] {len(self.samples)} pairs found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        deg_path, clean_path, task_id = self.samples[idx]
        deg   = Image.open(deg_path).convert("RGB")
        clean = Image.open(clean_path).convert("RGB")

        # Random crop
        i, j, h, w = T.RandomCrop.get_params(deg,
                         (self.patch_size, self.patch_size))
        deg   = TF.crop(deg,   i, j, h, w)
        clean = TF.crop(clean, i, j, h, w)

        # Augment
        if self.augment:
            if random.random() > 0.5:
                deg, clean = TF.hflip(deg), TF.hflip(clean)
            if random.random() > 0.5:
                deg, clean = TF.vflip(deg), TF.vflip(clean)

        deg   = TF.to_tensor(deg)
        clean = TF.to_tensor(clean)
        return deg, clean, task_id


# ─────────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model
    model = DFPIR(
        dim=args.dim,
        num_blocks=[4, 6, 6, 8],
        num_tasks=5,
        prompt_dim=args.prompt_dim,
        gamma=0.9,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model params: {total_params:.2f}M")

    # Dataset
    dataset = DegradedDataset(args.data_root, patch_size=args.patch_size)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=True, num_workers=4, pin_memory=True)

    # Optimizer + scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    criterion = nn.L1Loss()

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for deg, clean, task_id in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}"):
            deg      = deg.to(device)
            clean    = clean.to(device)
            task_id  = task_id.to(device)

            restored = model(deg, task_id=task_id)
            loss     = criterion(restored, clean)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        lr_now   = optimizer.param_groups[0]["lr"]
        print(f"  loss={avg_loss:.4f}  lr={lr_now:.2e}")

        if epoch % args.save_every == 0:
            ckpt = os.path.join(args.save_dir, f"dfpir_epoch{epoch:04d}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"  → saved {ckpt}")

    # Fine-tune stage (larger patches, smaller lr)
    print("\n[Fine-tune stage]")
    ft_dataset = DegradedDataset(args.data_root, patch_size=192)
    ft_loader  = DataLoader(ft_dataset, batch_size=3, shuffle=True,
                            num_workers=4, pin_memory=True)
    for param_group in optimizer.param_groups:
        param_group["lr"] = 1e-5

    for epoch in range(args.epochs + 1, args.epochs + 6):
        model.train()
        total_loss = 0.0
        for deg, clean, task_id in tqdm(ft_loader, desc=f"FT Epoch {epoch}"):
            deg     = deg.to(device)
            clean   = clean.to(device)
            task_id = task_id.to(device)
            restored = model(deg, task_id=task_id)
            loss = criterion(restored, clean)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  FT loss={total_loss/len(ft_loader):.4f}")

    final_ckpt = os.path.join(args.save_dir, "dfpir_final.pth")
    torch.save(model.state_dict(), final_ckpt)
    print(f"\n✓ Training complete. Model saved to {final_ckpt}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser("DFPIR Training")
    p.add_argument("--data_root",   required=True,
                   help="Root directory with task sub-folders")
    p.add_argument("--save_dir",    default="checkpoints")
    p.add_argument("--dim",         type=int,   default=48)
    p.add_argument("--prompt_dim",  type=int,   default=512)
    p.add_argument("--patch_size",  type=int,   default=128)
    p.add_argument("--batch_size",  type=int,   default=5)
    p.add_argument("--epochs",      type=int,   default=80)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--save_every",  type=int,   default=10)
    args = p.parse_args()
    train(args)
