"""
Quality Gate — Pre-screening images before model inference.

Checks:
  - Blur / motion blur (Laplacian variance)
  - Brightness (mean pixel intensity)
  - Resolution (minimum dimension + warning tier)
  - Glare / hotspot detection (overexposed pixel ratio)
  - Reflection detection (low-saturation hotspots + high local contrast)
  - Mud / dirt detection (desaturation in localised patches)
  - Colour extreme detection (very dark / very bright vehicles)
  - Matte paint detection (low specular variance)

Returns a structured QualityReport that the Enhancement module uses.
"""
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List

from pipeline.config import (
    BLUR_THRESHOLD, BLUR_SEVERE, BRIGHTNESS_LOW, BRIGHTNESS_HIGH,
    MIN_IMAGE_DIM, WARN_IMAGE_DIM,
    GLARE_HOTSPOT_THRESH, GLARE_HOTSPOT_RATIO, GLARE_SATURATION_LOW,
    LOCAL_CONTRAST_BLOCK, REFLECTION_CONTRAST_THRESH,
    SATURATION_LOW_THRESH, COLOR_EXTREME_DARK, COLOR_EXTREME_BRIGHT,
)


@dataclass
class QualityReport:
    """Result of the quality gate inspection."""
    passed: bool = True
    rejected: bool = False
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Measured metrics
    blur_score: float = 0.0
    brightness_mean: float = 0.0
    resolution: tuple = (0, 0)
    glare_ratio: float = 0.0
    saturation_mean: float = 0.0
    local_contrast_max: float = 0.0
    value_mean: float = 0.0

    # Enhancement flags (consumed by enhancement.py)
    needs_low_light_fix: bool = False
    needs_overexposure_fix: bool = False
    needs_denoise: bool = False
    needs_glare_reduction: bool = False
    needs_reflection_suppression: bool = False
    needs_contrast_normalisation: bool = False
    needs_upscale: bool = False

    # Detected conditions (informational, for downstream logic)
    is_color_extreme_dark: bool = False    # black car
    is_color_extreme_bright: bool = False  # white car
    is_matte_paint: bool = False
    has_mud_patches: bool = False
    has_water_droplets: bool = False       # flagged but not fixable

    # VLM context (populated by VLMContextEngine, if available)
    vlm_context: dict = field(default_factory=dict)

    def to_dict(self):
        result = {
            "passed": self.passed,
            "rejected": self.rejected,
            "issues": self.issues,
            "warnings": self.warnings,
            "blur_score": round(self.blur_score, 2),
            "brightness_mean": round(self.brightness_mean, 2),
            "glare_ratio": round(self.glare_ratio, 4),
            "saturation_mean": round(self.saturation_mean, 2),
            "resolution": list(self.resolution),
            "conditions": {
                "color_extreme_dark": self.is_color_extreme_dark,
                "color_extreme_bright": self.is_color_extreme_bright,
                "matte_paint": self.is_matte_paint,
                "mud_patches": self.has_mud_patches,
                "water_droplets": self.has_water_droplets,
            },
        }
        result["vlm_context"] = self.vlm_context
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Individual Assessment Functions
# ──────────────────────────────────────────────────────────────────────────────

def assess_blur(gray: np.ndarray) -> float:
    """Laplacian variance — low = blurry."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def assess_brightness(gray: np.ndarray) -> float:
    """Mean intensity of the grayscale image."""
    return float(np.mean(gray))


def assess_glare(hsv: np.ndarray) -> dict:
    """
    Detect glare hotspots: bright pixels with low saturation.
    Reflections on polished surfaces appear as bright, desaturated patches.
    """
    h, s, v = cv2.split(hsv)

    # Hotspot: very bright pixels
    hotspot_mask = v > GLARE_HOTSPOT_THRESH
    hotspot_ratio = float(hotspot_mask.sum()) / max(v.size, 1)

    # Reflection subset: hotspots with LOW saturation (not coloured objects)
    reflection_mask = hotspot_mask & (s < GLARE_SATURATION_LOW)
    reflection_ratio = float(reflection_mask.sum()) / max(v.size, 1)

    return {
        "hotspot_ratio": hotspot_ratio,
        "reflection_ratio": reflection_ratio,
        "has_glare": hotspot_ratio > GLARE_HOTSPOT_RATIO,
        "has_reflections": reflection_ratio > 0.05,
    }


def assess_local_contrast(gray: np.ndarray) -> float:
    """
    Compute max local contrast (block-wise std deviation).
    High local contrast in smooth regions = specular reflections.
    """
    block = LOCAL_CONTRAST_BLOCK
    h, w = gray.shape
    max_std = 0.0
    for y in range(0, h - block, block):
        for x in range(0, w - block, block):
            patch = gray[y:y+block, x:x+block].astype(np.float32)
            std = float(np.std(patch))
            if std > max_std:
                max_std = std
    return max_std


def assess_saturation(hsv: np.ndarray) -> dict:
    """
    Analyse saturation channel for mud / dirt (desaturated brown)
    and overall saturation health.
    """
    h_ch, s, v = cv2.split(hsv)

    mean_sat = float(np.mean(s))
    low_sat_ratio = float((s < SATURATION_LOW_THRESH).sum()) / max(s.size, 1)

    # Mud detection: low saturation + mid-value (brownish grey)
    mud_mask = (s < 35) & (v > 60) & (v < 180)
    mud_ratio = float(mud_mask.sum()) / max(s.size, 1)

    return {
        "mean_saturation": mean_sat,
        "low_sat_ratio": low_sat_ratio,
        "mud_ratio": mud_ratio,
        "has_mud_patches": mud_ratio > 0.25,  # >25% = likely mud/dirt
    }


def assess_color_extreme(hsv: np.ndarray) -> dict:
    """
    Detect color extremes: very dark (black car) or very bright (white car).
    These cars need contrast normalisation to avoid hiding damage.
    """
    v = hsv[:, :, 2]
    mean_v = float(np.mean(v))
    std_v = float(np.std(v))

    return {
        "mean_value": mean_v,
        "std_value": std_v,
        "is_dark": mean_v < COLOR_EXTREME_DARK,
        "is_bright": mean_v > COLOR_EXTREME_BRIGHT,
    }


def assess_matte_paint(gray: np.ndarray) -> bool:
    """
    Matte paint has low specular highlight variance.
    Gloss paint shows sharp bright reflections (high variance).
    Matte paint shows uniform low contrast.
    """
    # Compute variance of the top-10% brightest pixels
    sorted_pixels = np.sort(gray.ravel())
    top_10 = sorted_pixels[int(len(sorted_pixels) * 0.9):]
    top_var = float(np.var(top_10))

    # Matte: top highlights are very uniform (low variance)
    return top_var < 50  # threshold tuned down to prevent glossy cars flagging as matte


def assess_water_droplets(gray: np.ndarray) -> bool:
    """
    Detect water droplet patterns: many small bright circular features.
    Uses blob detection heuristic on the high-frequency component.
    """
    # High-pass filter to find small bright features
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    high_freq = cv2.absdiff(gray, blurred)

    # Threshold high-frequency bright spots
    _, thresh = cv2.threshold(high_freq, 25, 255, cv2.THRESH_BINARY)

    # Count small connected components (water droplet sized)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)

    # Filter: droplet-sized blobs (10-500 px area, roughly circular)
    droplet_count = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w_blob = stats[i, cv2.CC_STAT_WIDTH]
        h_blob = stats[i, cv2.CC_STAT_HEIGHT]
        if 10 < area < 500 and 0.3 < (w_blob / max(h_blob, 1)) < 3.0:
            droplet_count += 1

    return droplet_count > 50  # many small round features


# ──────────────────────────────────────────────────────────────────────────────
# File Integrity Check
# ──────────────────────────────────────────────────────────────────────────────

def is_image_corrupt(file_path: str) -> bool:
    """
    Check if an image file (e.g. JPEG) is physically truncated or corrupt.
    OpenCV's imread silently loads partial JPEGs (with 'premature end of data' warnings).
    PIL's verify() or attempting to fully load the image catches these structural errors.
    """
    try:
        from PIL import Image
        with Image.open(file_path) as img:
            img.verify()  # fast structural check
        # verify() doesn't always catch truncated data, so we attempt a full load
        with Image.open(file_path) as img:
            img.load()
        return False
    except Exception as e:
        return True


# ──────────────────────────────────────────────────────────────────────────────
# Main Quality Gate
# ──────────────────────────────────────────────────────────────────────────────

def run_quality_gate(image: np.ndarray, file_path: str = None) -> QualityReport:
    """
    Run all quality checks on a BGR numpy image.

    Args:
        image: np.ndarray of shape (H, W, 3), uint8, BGR.
        file_path: Optional path to the image for strict file integrity checking.

    Returns:
        QualityReport with detailed assessment and enhancement flags.
    """
    report = QualityReport()

    # ── 0. Strict File Integrity Check ────────────────────────────────────
    if file_path is not None and is_image_corrupt(file_path):
        report.rejected = True
        report.passed = False
        report.issues.append("Corrupt file: Premature end of data or unreadable format")
        return report

    h, w = image.shape[:2]
    report.resolution = (w, h)

    # Convert colour spaces once
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # ── 1. Resolution Check ───────────────────────────────────────────────
    min_dim = min(h, w)
    if min_dim < MIN_IMAGE_DIM:
        report.rejected = True
        report.passed = False
        report.issues.append(
            f"Resolution too low ({w}x{h}); minimum {MIN_IMAGE_DIM}px"
        )
        return report

    if min_dim < WARN_IMAGE_DIM:
        report.warnings.append(
            f"Low resolution ({w}x{h}); results may be less accurate"
        )
        report.needs_upscale = True

    # ── 2. Blur / Motion Blur ─────────────────────────────────────────────
    report.blur_score = assess_blur(gray)

    if report.blur_score < BLUR_SEVERE:
        report.passed = False
        report.issues.append(
            f"Severe motion blur (score={report.blur_score:.1f}, "
            f"reject_thresh={BLUR_SEVERE})"
        )
        # Don't reject — still processable but mark it
        report.needs_denoise = True

    elif report.blur_score < BLUR_THRESHOLD:
        report.passed = False
        report.needs_denoise = True
        report.warnings.append(
            f"Mild blur detected (score={report.blur_score:.1f})"
        )

    # ── 3. Brightness ─────────────────────────────────────────────────────
    report.brightness_mean = assess_brightness(gray)

    if report.brightness_mean < BRIGHTNESS_LOW:
        report.passed = False
        report.needs_low_light_fix = True
        report.issues.append(
            f"Low light (mean={report.brightness_mean:.0f})"
        )

    if report.brightness_mean > BRIGHTNESS_HIGH:
        report.passed = False
        report.needs_overexposure_fix = True
        report.issues.append(
            f"Overexposed (mean={report.brightness_mean:.0f})"
        )

    # ── 4. Glare & Reflections ────────────────────────────────────────────
    glare = assess_glare(hsv)
    report.glare_ratio = glare["hotspot_ratio"]

    if glare["has_glare"]:
        report.passed = False
        report.needs_glare_reduction = True
        report.issues.append(
            f"Glare detected ({report.glare_ratio:.1%} hotspot pixels)"
        )

    if glare["has_reflections"]:
        report.needs_reflection_suppression = True
        report.warnings.append(
            "Specular reflections detected — may cause false dent predictions"
        )

    # ── 5. Local Contrast (reflection patterns) ───────────────────────────
    report.local_contrast_max = assess_local_contrast(gray)
    if report.local_contrast_max > REFLECTION_CONTRAST_THRESH:
        report.needs_reflection_suppression = True
        report.warnings.append(
            f"High local contrast ({report.local_contrast_max:.0f}) — "
            f"possible specular highlights on polished surface"
        )

    # ── 6. Saturation Analysis (mud / dirt) ───────────────────────────────
    sat_info = assess_saturation(hsv)
    report.saturation_mean = sat_info["mean_saturation"]
    report.has_mud_patches = sat_info["has_mud_patches"]

    if report.has_mud_patches:
        report.warnings.append(
            f"Mud/dirt detected ({sat_info['mud_ratio']:.1%} area) — "
            f"may cause false scratch predictions"
        )

    # ── 7. Colour Extremes (black / white car) ───────────────────────────
    color = assess_color_extreme(hsv)
    report.value_mean = color["mean_value"]

    if color["is_dark"]:
        report.is_color_extreme_dark = True
        report.needs_contrast_normalisation = True
        report.warnings.append(
            f"Very dark vehicle (V={color['mean_value']:.0f}) — "
            f"damage may be hidden; applying contrast normalisation"
        )

    if color["is_bright"]:
        report.is_color_extreme_bright = True
        report.needs_contrast_normalisation = True
        report.warnings.append(
            f"Very bright vehicle (V={color['mean_value']:.0f}) — "
            f"subtle damage may blend with surface"
        )

    # ── 8. Matte Paint Detection ──────────────────────────────────────────
    report.is_matte_paint = assess_matte_paint(gray)
    if report.is_matte_paint:
        report.warnings.append(
            "Matte paint detected — reflections unlikely, "
            "texture-based damage detection prioritised"
        )

    # ── 9. Water Droplet Detection ────────────────────────────────────────
    report.has_water_droplets = assess_water_droplets(gray)
    if report.has_water_droplets:
        report.warnings.append(
            "Water droplets detected on surface — "
            "may interfere with scratch/crack predictions"
        )

    return report
