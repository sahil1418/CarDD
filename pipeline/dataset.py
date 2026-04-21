"""
Dataset — DualHeadDataset ported from the notebook + severity dataset.

DualHeadDataset:
  - Returns (image, damage_mask, anatomy_mask)
  - Masks use ignore_index=255 for missing supervision

SeverityDataset:
  - Folder-based (01-minor, 02-moderate, 03-severe)
  - Returns (image, label)

Augmentations are hardened for robustness against:
  - Glare / overexposure
  - Mud / dirt
  - Reflections / specular highlights
  - Water droplets
  - Motion blur
  - Low resolution / compression
  - Matte vs gloss paint
  - Colour extremes (black / white cars)
"""
import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from pipeline.quality_gate import is_image_corrupt
from pipeline.config import (
    IMG_SIZE, IMG_MEAN, IMG_STD, IGNORE_INDEX,
    SEVERITY_DATA_DIR, SEVERITY_CLASSES,
)


# ──────────────────────────────────────────────────────────────────────────────
# Custom Augmentation: Simulated Mud / Dirt Splatter
# ──────────────────────────────────────────────────────────────────────────────

class SimulateMud(A.ImageOnlyTransform):
    """
    Overlay random brownish-grey patches to simulate mud/dirt.
    Teaches the model NOT to confuse mud with scratches or paint damage.
    """

    def __init__(self, num_patches=(3, 12), patch_size=(20, 80),
                 always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.num_patches = num_patches
        self.patch_size = patch_size

    def apply(self, img, **params):
        result = img.copy()
        h, w = result.shape[:2]
        n = np.random.randint(*self.num_patches)

        for _ in range(n):
            # Random brown/grey color (mud range)
            color = np.array([
                np.random.randint(80, 140),   # R
                np.random.randint(60, 110),   # G
                np.random.randint(40, 80),    # B
            ], dtype=np.uint8)

            # Random ellipse-shaped splatter
            cx = np.random.randint(0, w)
            cy = np.random.randint(0, h)
            ax1 = np.random.randint(*self.patch_size)
            ax2 = np.random.randint(self.patch_size[0] // 2,
                                    self.patch_size[1])

            mask = np.zeros((h, w), dtype=np.uint8)
            angle = np.random.randint(0, 180)
            cv2.ellipse(mask, (cx, cy), (ax1, ax2), angle, 0, 360, 255, -1)

            # Blur the mask for soft edges
            mask = cv2.GaussianBlur(mask, (15, 15), 0)
            alpha = (mask.astype(np.float32) / 255.0) * np.random.uniform(0.3, 0.7)
            alpha = alpha[..., np.newaxis]

            color_layer = np.full_like(result, color)
            result = (result * (1 - alpha) + color_layer * alpha).astype(np.uint8)

        return result

    def get_transform_init_args_names(self):
        return ("num_patches", "patch_size")


# ──────────────────────────────────────────────────────────────────────────────
# Custom Augmentation: Simulated Water Droplets
# ──────────────────────────────────────────────────────────────────────────────

class SimulateWaterDroplets(A.ImageOnlyTransform):
    """
    Add random bright circular spots to simulate water droplets.
    Prevents false scratch/crack predictions on wet surfaces.
    """

    def __init__(self, num_drops=(10, 60), radius_range=(2, 8),
                 always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.num_drops = num_drops
        self.radius_range = radius_range

    def apply(self, img, **params):
        result = img.copy().astype(np.float32)
        h, w = result.shape[:2]
        n = np.random.randint(*self.num_drops)

        for _ in range(n):
            cx = np.random.randint(0, w)
            cy = np.random.randint(0, h)
            r = np.random.randint(*self.radius_range)

            # Create bright highlight (water refraction effect)
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.circle(mask, (cx, cy), r, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (r * 2 + 1, r * 2 + 1), 0)

            brightness_boost = np.random.uniform(30, 80)
            result += mask[..., np.newaxis] * brightness_boost

        return np.clip(result, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("num_drops", "radius_range")


# ──────────────────────────────────────────────────────────────────────────────
# Custom Augmentation: Specular Highlight Simulation
# ──────────────────────────────────────────────────────────────────────────────

class SimulateSpecularHighlight(A.ImageOnlyTransform):
    """
    Add large bright specular reflections (simulating polished car surface
    under strong light). Teaches model to ignore reflections as non-damage.
    """

    def __init__(self, num_highlights=(1, 3),
                 always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.num_highlights = num_highlights

    def apply(self, img, **params):
        result = img.copy().astype(np.float32)
        h, w = result.shape[:2]
        n = np.random.randint(*self.num_highlights)

        for _ in range(n):
            cx = np.random.randint(w // 4, 3 * w // 4)
            cy = np.random.randint(h // 4, 3 * h // 4)
            ax1 = np.random.randint(30, w // 3)
            ax2 = np.random.randint(20, h // 4)
            angle = np.random.randint(0, 180)

            mask = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(mask, (cx, cy), (ax1, ax2), angle, 0, 360, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (51, 51), 0)

            # Bright white highlight
            intensity = np.random.uniform(60, 150)
            result += mask[..., np.newaxis] * intensity

        return np.clip(result, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("num_highlights",)


# ──────────────────────────────────────────────────────────────────────────────
# Custom Augmentation: Colour Shift for Vehicle Colour Extremes
# ──────────────────────────────────────────────────────────────────────────────

class SimulateColorExtreme(A.ImageOnlyTransform):
    """
    Randomly shift image to simulate very dark (black) or very bright (white)
    vehicles. Forces the model to learn damage features across colour range.
    """

    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        result = img.copy()
        mode = np.random.choice(["darken", "brighten"])

        if mode == "darken":
            factor = np.random.uniform(0.2, 0.5)
            result = (result.astype(np.float32) * factor).astype(np.uint8)
        else:
            factor = np.random.uniform(1.5, 2.5)
            result = np.clip(
                result.astype(np.float32) * factor, 0, 255
            ).astype(np.uint8)

        return result

    def get_transform_init_args_names(self):
        return ()


# ──────────────────────────────────────────────────────────────────────────────
# Safe Albumentations constructors (compatible with 1.x, 2.0, and 2.1+)
#
# Strategy: try newest API first → catch (TypeError | ValueError) → fallback.
# Albumentations 2.1+ uses pydantic and rejects unknown params with ValueError.
# ──────────────────────────────────────────────────────────────────────────────

def _safe_gauss_noise(p=1.0):
    """GaussNoise: var_limit (1.x) → std_range (2.x)."""
    try:
        return A.GaussNoise(std_range=(0.02, 0.08), p=p)
    except (TypeError, ValueError):
        return A.GaussNoise(var_limit=(10, 60), p=p)


def _safe_image_compression(p=0.15):
    """ImageCompression: quality_lower/upper (1.x) → quality_range (2.x)."""
    try:
        return A.ImageCompression(quality_range=(40, 95), p=p)
    except (TypeError, ValueError):
        return A.ImageCompression(quality_lower=40, quality_upper=95, p=p)


def _safe_downscale(p=0.1):
    """Downscale: scale_min/max (1.x) → scale_range (2.x)."""
    try:
        return A.Downscale(scale_range=(0.25, 0.5), p=p)
    except (TypeError, ValueError):
        try:
            return A.Downscale(scale_min=0.25, scale_max=0.5, p=p)
        except (TypeError, ValueError):
            return A.Downscale(
                scale_min=0.25, scale_max=0.5,
                interpolation=cv2.INTER_LINEAR, p=p
            )


def _safe_random_rain(p=0.1):
    """RandomRain: rain_type is required in 2.1+ (cannot be None)."""
    # Try newest API (minimal params + explicit rain_type)
    try:
        return A.RandomRain(rain_type="drizzle", p=p)
    except (TypeError, ValueError):
        pass
    # Try older API with full params
    try:
        return A.RandomRain(
            slant_lower=-10, slant_upper=10,
            drop_length=15, drop_width=1, drop_color=(200, 200, 200),
            blur_value=3, brightness_coefficient=0.9, rain_type=None,
            p=p
        )
    except (TypeError, ValueError):
        return A.RandomRain(p=p)


def _safe_random_sun_flare(p=0.08):
    """RandomSunFlare: param names changed across versions."""
    # Try minimal (2.1+ compatible)
    try:
        return A.RandomSunFlare(p=p)
    except (TypeError, ValueError):
        pass
    # Try full old params
    try:
        return A.RandomSunFlare(
            flare_roi=(0, 0, 1, 0.5),
            angle_lower=0, angle_upper=1,
            num_flare_circles_lower=3, num_flare_circles_upper=6,
            src_radius=100, src_color=(255, 255, 255), p=p
        )
    except (TypeError, ValueError):
        return A.RandomSunFlare(p=p)


def _safe_random_fog(p=0.1):
    """RandomFog: fog_coef_lower/upper (1.x) → fog_coef_range (2.x)."""
    # Try new API
    try:
        return A.RandomFog(fog_coef_range=(0.1, 0.3), alpha_coef=0.1, p=p)
    except (TypeError, ValueError):
        pass
    # Try old API
    try:
        return A.RandomFog(
            fog_coef_lower=0.1, fog_coef_upper=0.3,
            alpha_coef=0.1, p=p
        )
    except (TypeError, ValueError):
        return A.RandomFog(p=p)


def _safe_random_shadow(p=0.2):
    """RandomShadow: num_shadows_lower/upper (1.x) → num_shadows_limit (2.x)."""
    # Try new API
    try:
        return A.RandomShadow(
            shadow_roi=(0, 0, 1, 1),
            num_shadows_limit=(1, 3),
            shadow_dimension=6, p=p
        )
    except (TypeError, ValueError):
        pass
    # Try old API
    try:
        return A.RandomShadow(
            shadow_roi=(0, 0, 1, 1),
            num_shadows_lower=1, num_shadows_upper=3,
            shadow_dimension=6, p=p
        )
    except (TypeError, ValueError):
        return A.RandomShadow(p=p)


def _safe_pad_if_needed(size, border_mode=cv2.BORDER_CONSTANT):
    """PadIfNeeded: 'value' param removed in 2.1+."""
    try:
        return A.PadIfNeeded(size, size, border_mode=border_mode, fill=0)
    except (TypeError, ValueError):
        return A.PadIfNeeded(size, size, border_mode=border_mode, value=0)


def _safe_shift_scale_rotate(p=0.3):
    """ShiftScaleRotate deprecated in 2.x → use Affine."""
    try:
        return A.Affine(
            translate_percent=(-0.05, 0.05),
            scale=(0.85, 1.15),
            rotate=(-15, 15),
            border_mode=cv2.BORDER_CONSTANT, p=p
        )
    except (TypeError, ValueError):
        pass
    try:
        return A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.15, rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT, p=p
        )
    except (TypeError, ValueError):
        return A.Affine(p=p)


# ──────────────────────────────────────────────────────────────────────────────
# Training Augmentations (hardened for all edge cases)
# ──────────────────────────────────────────────────────────────────────────────

def get_train_transform():
    return A.Compose([
        # ── Spatial ───────────────────────────────────────────────────────
        A.LongestMaxSize(IMG_SIZE),
        _safe_pad_if_needed(IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        _safe_shift_scale_rotate(p=0.3),

        # ── Colour & Light (glare, overexposure, low-light) ──────────────
        A.RandomBrightnessContrast(
            brightness_limit=0.4, contrast_limit=0.4, p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=15, sat_shift_limit=40,
            val_shift_limit=40, p=0.4),
        A.RandomGamma(gamma_limit=(60, 150), p=0.3),
        A.RandomToneCurve(scale=0.15, p=0.2),

        # ── Environmental Robustness ─────────────────────────────────────
        _safe_random_shadow(),
        _safe_random_fog(),
        _safe_random_rain(),
        _safe_random_sun_flare(),

        # ── Noise & Blur (motion blur, compression, sensor noise) ────────
        A.OneOf([
            _safe_gauss_noise(),
            A.ISONoise(color_shift=(0.01, 0.05),
                       intensity=(0.1, 0.5), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
        ], p=0.35),

        A.OneOf([
            A.MotionBlur(blur_limit=(3, 9), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.2),

        _safe_image_compression(),

        # ── Custom Edge-Case Augmentations ────────────────────────────────
        SimulateMud(num_patches=(3, 10), patch_size=(15, 60), p=0.12),
        SimulateWaterDroplets(num_drops=(10, 40), radius_range=(2, 6), p=0.10),
        SimulateSpecularHighlight(num_highlights=(1, 2), p=0.10),
        SimulateColorExtreme(p=0.08),

        # ── Contrast Normalisation (helps with matte/gloss variance) ─────
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.2),
        A.Equalize(p=0.05),

        # ── Low Resolution Simulation ────────────────────────────────────
        _safe_downscale(),

        # ── Final Normalisation ──────────────────────────────────────────
        A.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ToTensorV2(),
    ], additional_targets={"anatomy_mask": "mask"})


def get_val_transform():
    return A.Compose([
        A.LongestMaxSize(IMG_SIZE),
        _safe_pad_if_needed(IMG_SIZE),
        A.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ToTensorV2(),
    ], additional_targets={"anatomy_mask": "mask"})


# ──────────────────────────────────────────────────────────────────────────────
# Dual-Head Dataset (Damage + Anatomy)
# ──────────────────────────────────────────────────────────────────────────────

class DualHeadDataset(Dataset):
    """
    Reads manifests produced by the notebook's registry system.

    Each item has:
        image_path, hash, has_damage, has_anatomy, has_severity,
        damage_classes, dataset

    Annotation polygons are rasterised on-the-fly from the GlobalAnnotationDB.
    """

    def __init__(self, manifest_path: str, annotation_db,
                 transform=None):
        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)
        self.annotation_db = annotation_db
        self.transform = transform

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        item = self.manifest[idx]
        image_path = item["image_path"]
        img_hash = item["hash"]

        # Load image safely
        if is_image_corrupt(image_path):
            image = None
        else:
            image = cv2.imread(image_path)

        if image is None:
            # Fallback: return a dummy sample (ignored in loss via IGNORE_INDEX)
            image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        damage_mask  = np.zeros((h, w), dtype=np.uint8)
        anatomy_mask = np.zeros((h, w), dtype=np.uint8)

        # Rasterize damage polygons
        if item["has_damage"]:
            for poly in self.annotation_db.damage_db.get(img_hash, []):
                pts = np.array(poly["points"], dtype=np.int32)
                cv2.fillPoly(damage_mask, [pts], poly["class_id"])
            anatomy_mask.fill(IGNORE_INDEX)

        # Rasterize anatomy polygons
        elif item["has_anatomy"]:
            for poly in self.annotation_db.anatomy_db.get(img_hash, []):
                pts = np.array(poly["points"], dtype=np.int32)
                cv2.fillPoly(anatomy_mask, [pts], poly["class_id"])
            damage_mask.fill(IGNORE_INDEX)

        # Normal images → both are background (0), no ignore
        # This is the default from zeros init.

        # Apply augmentations
        if self.transform:
            augmented = self.transform(
                image=image,
                mask=damage_mask,
                anatomy_mask=anatomy_mask,
            )
            image        = augmented["image"]
            damage_mask  = augmented["mask"].long()
            anatomy_mask = augmented["anatomy_mask"].long()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            damage_mask  = torch.from_numpy(damage_mask).long()
            anatomy_mask = torch.from_numpy(anatomy_mask).long()

        return image, damage_mask, anatomy_mask


# ──────────────────────────────────────────────────────────────────────────────
# Severity Dataset (folder-based classification)
# ──────────────────────────────────────────────────────────────────────────────

class SeverityDataset(Dataset):
    """
    Folder structure:
        data3a/
          training/
            01-minor/    → label 0
            02-moderate/ → label 1
            03-severe/   → label 2
          validation/
            01-minor/ ...
    """

    def __init__(self, root_dir: str, split: str = "training",
                 transform=None):
        self.samples = []
        self.transform = transform

        split_dir = os.path.join(root_dir, split)
        if not os.path.exists(split_dir):
            # Try flat structure
            split_dir = root_dir

        for folder_name in sorted(os.listdir(split_dir)):
            folder_path = os.path.join(split_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            # Map folder to severity label
            label = None
            lower = folder_name.lower()
            if "minor" in lower:
                label = SEVERITY_CLASSES["minor"]
            elif "moderate" in lower:
                label = SEVERITY_CLASSES["moderate"]
            elif "severe" in lower:
                label = SEVERITY_CLASSES["severe"]

            if label is None:
                continue

            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((
                        os.path.join(folder_path, img_name), label
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        if is_image_corrupt(path):
            image = None
        else:
            image = cv2.imread(path)
            
        if image is None:
            image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, label


def get_severity_transform(train: bool = True):
    """Severity classification augmentations — also hardened."""
    if train:
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=0.4),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=20,
                val_shift_limit=20, p=0.3),
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=5, p=1.0),
            ], p=0.15),
            _safe_gauss_noise(p=0.2),
            _safe_image_compression(p=0.1),
            SimulateColorExtreme(p=0.05),
            A.Normalize(mean=IMG_MEAN, std=IMG_STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ToTensorV2(),
    ])