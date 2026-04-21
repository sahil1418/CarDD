"""
Conditional Enhancement — Applied when Quality Gate flags issues.

Strategies (expanded for robustness):
  - Low light       → CLAHE on L-channel (LAB)
  - Overexposed     → Gamma correction
  - Noisy / blur    → Edge-preserving bilateral filter
  - Glare           → Adaptive highlight compression + local tone mapping
  - Reflections     → Saturation-guided despekularing
  - Contrast norm   → Normalise contrast for black/white cars
  - Low resolution  → INTER_LANCZOS4 upscale to 512×512
"""
import cv2
import numpy as np

from pipeline.quality_gate import QualityReport
from pipeline.config import IMG_SIZE


# ──────────────────────────────────────────────────────────────────────────────
# Core Enhancement Functions
# ──────────────────────────────────────────────────────────────────────────────

def apply_clahe(image: np.ndarray, clip_limit: float = 3.0,
                grid_size: int = 8) -> np.ndarray:
    """Adaptive histogram equalization on the L channel (LAB)."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                            tileGridSize=(grid_size, grid_size))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def apply_gamma(image: np.ndarray, gamma: float = 0.6) -> np.ndarray:
    """Gamma correction — gamma < 1 darkens (fixes overexposure)."""
    inv = 1.0 / gamma
    table = np.array(
        [(i / 255.0) ** inv * 255 for i in range(256)], dtype=np.uint8
    )
    return cv2.LUT(image, table)


def apply_bilateral(image: np.ndarray, d: int = 9,
                    sigma_color: float = 75,
                    sigma_space: float = 75) -> np.ndarray:
    """Edge-preserving denoising for motion blur / noise."""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


# ──────────────────────────────────────────────────────────────────────────────
# NEW: Glare Reduction
# ──────────────────────────────────────────────────────────────────────────────

def apply_glare_reduction(image: np.ndarray) -> np.ndarray:
    """
    Reduce glare by compressing highlights in the V channel.
    Uses adaptive thresholding to target only the hotspot regions.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Identify glare hotspots
    hotspot_mask = v > 240

    if hotspot_mask.any():
        # Compress highlight values (soft clamp)
        v_float = v.astype(np.float32)

        # Apply tone curve: compress values > 200 towards 200
        high_mask = v_float > 200
        v_float[high_mask] = 200 + (v_float[high_mask] - 200) * 0.3

        v = np.clip(v_float, 0, 255).astype(np.uint8)

    # Also apply local tone mapping to even out intensity
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    v = clahe.apply(v)

    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)


# ──────────────────────────────────────────────────────────────────────────────
# NEW: Reflection Suppression
# ──────────────────────────────────────────────────────────────────────────────

def apply_reflection_suppression(image: np.ndarray) -> np.ndarray:
    """
    Suppress specular reflections on polished car surfaces.
    Reflections appear as bright, low-saturation spots.
    Strategy: boost saturation in reflection regions + tone-map V.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Detect reflection pixels: bright + low saturation
    reflection_mask = (v > 200) & (s < 30)

    if reflection_mask.any():
        # Reduce brightness in reflection areas
        v_float = v.astype(np.float32)
        v_float[reflection_mask] *= 0.7
        v = np.clip(v_float, 0, 255).astype(np.uint8)

        # Boost saturation slightly so underlying colour shows through
        s_float = s.astype(np.float32)
        s_float[reflection_mask] = np.minimum(s_float[reflection_mask] * 2.0 + 10, 255)
        s = np.clip(s_float, 0, 255).astype(np.uint8)

    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)


# ──────────────────────────────────────────────────────────────────────────────
# NEW: Contrast Normalisation (for black / white cars)
# ──────────────────────────────────────────────────────────────────────────────

def apply_contrast_normalisation(image: np.ndarray,
                                 is_dark: bool = False,
                                 is_bright: bool = False) -> np.ndarray:
    """
    Normalise contrast for colour-extreme vehicles.

    Dark cars:  boost mid-tones and local contrast so damage is visible.
    White cars: suppress highlights and boost micro-contrast in light areas.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    if is_dark:
        # Aggressively boost contrast on dark vehicles
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Additional gamma boost to lift shadows
        l_float = (l.astype(np.float32) / 255.0) ** 0.6 * 255
        l = np.clip(l_float, 0, 255).astype(np.uint8)

    elif is_bright:
        # Gentle contrast boost — don't over-darken
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Slight gamma darken to show subtle scratches/dents
        l_float = (l.astype(np.float32) / 255.0) ** 1.3 * 255
        l = np.clip(l_float, 0, 255).astype(np.uint8)

    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ──────────────────────────────────────────────────────────────────────────────
# NEW: Smart Upscale for Low Resolution
# ──────────────────────────────────────────────────────────────────────────────

def apply_smart_upscale(image: np.ndarray,
                        target_size: int = IMG_SIZE) -> np.ndarray:
    """
    Upscale low-resolution images using Lanczos interpolation.
    Only upscales the shorter edge to target_size while maintaining aspect.
    """
    h, w = image.shape[:2]
    scale = target_size / min(h, w)

    if scale <= 1.0:
        return image    # no upscale needed

    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


# ──────────────────────────────────────────────────────────────────────────────
# Main Enhancement Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def enhance_image(image: np.ndarray,
                  quality_report: QualityReport) -> np.ndarray:
    """
    Conditionally enhance the image based on Quality Gate findings.

    Enhancements are applied in a specific order to avoid conflicts:
      1. Upscale (if low-res)
      2. Glare reduction (before brightness fixes)
      3. Reflection suppression
      4. Low-light fix (CLAHE)
      5. Overexposure fix (gamma)
      6. Contrast normalisation (black/white car)
      7. Denoise (bilateral — always last to smooth artifacts)

    Args:
        image:          BGR uint8 numpy array (H, W, 3).
        quality_report: Output of run_quality_gate().

    Returns:
        Enhanced image (BGR uint8).
    """
    if quality_report.rejected:
        return image

    # If nothing is flagged, return unchanged
    needs_any = (
        quality_report.needs_upscale or
        quality_report.needs_glare_reduction or
        quality_report.needs_reflection_suppression or
        quality_report.needs_low_light_fix or
        quality_report.needs_overexposure_fix or
        quality_report.needs_contrast_normalisation or
        quality_report.needs_denoise
    )
    if not needs_any:
        return image

    result = image.copy()

    # 1. Upscale first (operate on higher resolution for better results)
    if quality_report.needs_upscale:
        result = apply_smart_upscale(result)

    # 2. Glare reduction (must come before general brightness fixes)
    if quality_report.needs_glare_reduction:
        result = apply_glare_reduction(result)

    # 3. Reflection suppression
    if quality_report.needs_reflection_suppression:
        result = apply_reflection_suppression(result)

    # 4. Low-light fix
    if quality_report.needs_low_light_fix:
        result = apply_clahe(result, clip_limit=4.0)

    # 5. Overexposure fix
    if quality_report.needs_overexposure_fix:
        result = apply_gamma(result, gamma=0.5)

    # 6. Contrast normalisation for colour extremes
    if quality_report.needs_contrast_normalisation:
        result = apply_contrast_normalisation(
            result,
            is_dark=quality_report.is_color_extreme_dark,
            is_bright=quality_report.is_color_extreme_bright,
        )

    # 7. Denoise — always last to smooth out artifacts from prior steps
    if quality_report.needs_denoise:
        result = apply_bilateral(result)

    return result
