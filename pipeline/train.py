"""
Training Pipeline — Two-stage training on Lightning AI L4 GPU.

Stage 1: Dual-head segmentation (DINOv2 + Mask2Former)
  - FP16 mixed precision
  - Gradient accumulation (effective batch = 16)
  - Class-weighted CE + Dice loss
  - CosineAnnealing + warmup

Stage 2: Severity classifier (EfficientNet-B0)
  - Standard classification training
"""
import os
import sys
import time
import math
import json
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from pipeline.config import (
    BASE, TRAIN_MANIFEST, VAL_MANIFEST, CHECKPOINT_DIR,
    STAGE1_EPOCHS, STAGE1_BATCH_SIZE, STAGE1_GRAD_ACCUM,
    STAGE1_LR_DECODER, STAGE1_LR_FPN, STAGE1_WEIGHT_DECAY,
    STAGE1_WARMUP_EPOCHS, STAGE1_LOSS_CE_W, STAGE1_LOSS_DICE_W,
    STAGE2_EPOCHS, STAGE2_BATCH_SIZE, STAGE2_LR,
    DAMAGE_CLASS_WEIGHTS, NUM_DAMAGE_CLASSES, NUM_ANATOMY_CLASSES,
    IGNORE_INDEX, IMG_SIZE, SEVERITY_DATA_DIR,
)
from pipeline.dataset import (
    DualHeadDataset, SeverityDataset,
    get_train_transform, get_val_transform, get_severity_transform,
)
from pipeline.models.backbone import DINOv2Backbone
from pipeline.models.mask2former import DualMask2Former
from pipeline.models.severity_model import SeverityClassifier


# ──────────────────────────────────────────────────────────────────────────────
# Losses
# ──────────────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """Per-class Dice loss with ignore_index support."""

    def __init__(self, num_classes: int, ignore_index: int = IGNORE_INDEX,
                 smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        logits:  (B, C, H, W)
        targets: (B, H, W) with class indices
        """
        probs = F.softmax(logits, dim=1)                    # (B, C, H, W)

        # Create valid mask
        valid = (targets != self.ignore_index)               # (B, H, W)
        targets_clean = targets.clone()
        targets_clean[~valid] = 0                            # safe for one_hot

        # One-hot encoding
        one_hot = F.one_hot(targets_clean, self.num_classes) # (B, H, W, C)
        one_hot = one_hot.permute(0, 3, 1, 2).float()       # (B, C, H, W)

        # Zero out invalid positions
        valid_4d = valid.unsqueeze(1).float()                # (B, 1, H, W)
        one_hot = one_hot * valid_4d
        probs   = probs * valid_4d

        # Per-class dice
        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dims)
        cardinality  = probs.sum(dims) + one_hot.sum(dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class CombinedSegLoss(nn.Module):
    """CE + Dice weighted loss for segmentation."""

    def __init__(self, num_classes: int, class_weights=None,
                 ce_weight: float = STAGE1_LOSS_CE_W,
                 dice_weight: float = STAGE1_LOSS_DICE_W):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        weight_tensor = None
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

        self.ce_loss   = nn.CrossEntropyLoss(
            weight=weight_tensor, ignore_index=IGNORE_INDEX
        )
        self.dice_loss = DiceLoss(num_classes)

    def forward(self, logits, targets):
        ce   = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.ce_weight * ce + self.dice_weight * dice


# ──────────────────────────────────────────────────────────────────────────────
# Training Helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def warmup_cosine_lr(optimizer, epoch, warmup_epochs, total_epochs, base_lr):
    """Linear warmup → cosine decay."""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr * pg.get("lr_scale", 1.0)


def gpu_mem_mb():
    """Current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: Dual-Head Segmentation Training
# ──────────────────────────────────────────────────────────────────────────────

def train_stage1(annotation_db, resume_ckpt: str = None):
    """
    Train the dual-head segmentation model.

    Args:
        annotation_db: GlobalAnnotationDB instance from your notebook.
        resume_ckpt:   path to checkpoint to resume from (optional).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Stage 1 — Device: {device}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────
    train_ds = DualHeadDataset(TRAIN_MANIFEST, annotation_db,
                               transform=get_train_transform())
    val_ds   = DualHeadDataset(VAL_MANIFEST, annotation_db,
                               transform=get_val_transform())

    train_loader = DataLoader(
        train_ds, batch_size=STAGE1_BATCH_SIZE,
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=STAGE1_BATCH_SIZE,
        shuffle=False, num_workers=4, pin_memory=True
    )

    # ── Models ────────────────────────────────────────────────────────────
    backbone = DINOv2Backbone(freeze=True).to(device)
    decoder  = DualMask2Former().to(device)

    # ── Loss ──────────────────────────────────────────────────────────────
    damage_loss_fn  = CombinedSegLoss(
        NUM_DAMAGE_CLASSES, class_weights=DAMAGE_CLASS_WEIGHTS
    ).to(device)
    anatomy_loss_fn = CombinedSegLoss(NUM_ANATOMY_CLASSES).to(device)

    # ── Optimizer (decoder + FPN only) ────────────────────────────────────
    param_groups = [
        {"params": backbone.fpn.parameters(),
         "lr": STAGE1_LR_FPN, "lr_scale": STAGE1_LR_FPN / STAGE1_LR_DECODER},
        {"params": decoder.parameters(),
         "lr": STAGE1_LR_DECODER, "lr_scale": 1.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=STAGE1_LR_DECODER,
                                   weight_decay=STAGE1_WEIGHT_DECAY)
    scaler = GradScaler('cuda')

    start_epoch = 0
    best_val_loss = float("inf")

    # Resume
    if resume_ckpt and os.path.exists(resume_ckpt):
        ckpt = torch.load(resume_ckpt, map_location=device)
        backbone.fpn.load_state_dict(ckpt["fpn_state"])
        decoder.load_state_dict(ckpt["decoder_state"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}")

    # ── Training Loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, STAGE1_EPOCHS):
        warmup_cosine_lr(optimizer, epoch, STAGE1_WARMUP_EPOCHS,
                         STAGE1_EPOCHS, STAGE1_LR_DECODER)
        backbone.train()        # FPN is trainable
        decoder.train()

        epoch_loss = 0.0
        optimizer.zero_grad()

        t0 = time.time()
        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch}/{STAGE1_EPOCHS}",
            bar_format="{l_bar}{bar:30}{r_bar}",
            dynamic_ncols=True,
        )
        for step, (images, dmg_masks, anat_masks) in pbar:
            images    = images.to(device)
            dmg_masks = dmg_masks.to(device)
            anat_masks= anat_masks.to(device)

            with autocast('cuda'):
                fpn_feats = backbone(images)
                dmg_logits, anat_logits, _ = decoder(fpn_feats)

                loss_dmg  = damage_loss_fn(dmg_logits, dmg_masks)
                loss_anat = anatomy_loss_fn(anat_logits, anat_masks)
                loss = (loss_dmg + loss_anat) / STAGE1_GRAD_ACCUM

            # Skip corrupt/NaN batches
                        # Skip corrupt/NaN batches
                if torch.isnan(loss) or torch.isinf(loss):
                    optimizer.zero_grad()
                    continue

                scaler.scale(loss).backward()

                # Only clip and step at the accumulation boundary
                if (step + 1) % STAGE1_GRAD_ACCUM == 0 or (step + 1) == len(train_loader):

                    # Clip SCALED gradients (scale max_norm by current scale factor)
                    # This avoids calling scaler.unscale_() entirely, eliminating
                    # the "unscale_() already called" RuntimeError forever.
                    scale = scaler.get_scale()
                    torch.nn.utils.clip_grad_norm_(
                        list(backbone.fpn.parameters()) + list(decoder.parameters()),
                        max_norm=1.0 * scale
                    )

                    # scaler.step() internally unscales before applying to optimizer
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()


            batch_loss = loss.item() * STAGE1_GRAD_ACCUM
            epoch_loss += batch_loss

            pbar.set_postfix({
                "loss": f"{batch_loss:.4f}",
                "lr": f"{get_lr(optimizer):.1e}",
                "VRAM": f"{gpu_mem_mb():.0f}MB",
            })

        epoch_loss /= len(train_loader)
        elapsed = time.time() - t0

        # ── Validation ────────────────────────────────────────────────────
        backbone.eval()
        decoder.eval()
        val_loss = 0.0

        with torch.no_grad():
            val_pbar = tqdm(
                val_loader, desc="  Validating",
                bar_format="{l_bar}{bar:30}{r_bar}",
                dynamic_ncols=True, leave=False,
            )
            for images, dmg_masks, anat_masks in val_pbar:
                images    = images.to(device)
                dmg_masks = dmg_masks.to(device)
                anat_masks= anat_masks.to(device)

                with autocast('cuda'):
                    fpn_feats = backbone(images)
                    dmg_logits, anat_logits, _ = decoder(fpn_feats)
                    loss_dmg  = damage_loss_fn(dmg_logits, dmg_masks)
                    loss_anat = anatomy_loss_fn(anat_logits, anat_masks)
                    loss = loss_dmg + loss_anat

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch}/{STAGE1_EPOCHS} — "
              f"train_loss={epoch_loss:.4f} val_loss={val_loss:.4f} "
              f"time={elapsed:.0f}s")

        # ── Save checkpoint ───────────────────────────────────────────────
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        ckpt = {
            "epoch": epoch,
            "fpn_state": backbone.fpn.state_dict(),
            "decoder_state": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": epoch_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
        }
        torch.save(ckpt, os.path.join(CHECKPOINT_DIR, "last.pt"))
        if is_best:
            torch.save(ckpt, os.path.join(CHECKPOINT_DIR, "best.pt"))
            print(f"  ✓ New best model (val_loss={val_loss:.4f})")

    print("Stage 1 complete!")
    return backbone, decoder


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: Severity Classifier Training
# ──────────────────────────────────────────────────────────────────────────────

def train_stage2():
    """Train the EfficientNet-B0 severity classifier."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Stage 2 — Device: {device}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_ds = SeverityDataset(SEVERITY_DATA_DIR, "training",
                                get_severity_transform(train=True))
    val_ds   = SeverityDataset(SEVERITY_DATA_DIR, "validation",
                                get_severity_transform(train=False))

    print(f"Severity train: {len(train_ds)}, val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=STAGE2_BATCH_SIZE,
                               shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=STAGE2_BATCH_SIZE,
                               shuffle=False, num_workers=4, pin_memory=True)

    model = SeverityClassifier(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=STAGE2_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=STAGE2_EPOCHS
    )

    best_acc = 0.0

    for epoch in range(STAGE2_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{STAGE2_EPOCHS}",
            bar_format="{l_bar}{bar:30}{r_bar}",
            dynamic_ncols=True,
        )
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100.0*correct/total:.1f}%",
            })

        scheduler.step()
        train_acc = 100.0 * correct / total
        train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / max(val_total, 1)

        print(f"Epoch {epoch}/{STAGE2_EPOCHS} — "
              f"loss={train_loss:.4f} train_acc={train_acc:.1f}% "
              f"val_acc={val_acc:.1f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINT_DIR, "severity_best.pt"))
            print(f"  ✓ New best severity model (acc={val_acc:.1f}%)")

    print(f"Stage 2 complete! Best val acc: {best_acc:.1f}%")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Usage (on Lightning AI):
        python -m pipeline.train --stage 1   # needs GlobalAnnotationDB
        python -m pipeline.train --stage 2   # severity classifier
        python -m pipeline.train --stage all  # both
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=["1", "2", "all"])
    parser.add_argument("--resume", default=None,
                        help="Checkpoint path to resume Stage 1")
    args = parser.parse_args()

    if args.stage in ("1", "all"):
        # Import and build GlobalAnnotationDB from notebook code
        print("Building GlobalAnnotationDB...")
        sys.path.insert(0, BASE)

        # The annotation DB setup is in the notebook cells.
        # For standalone usage, we inline a minimal version here.
        from pipeline._annotation_db import GlobalAnnotationDB
        global_db = GlobalAnnotationDB([TRAIN_MANIFEST, VAL_MANIFEST])
        train_stage1(global_db, resume_ckpt=args.resume)

    if args.stage in ("2", "all"):
        train_stage2()