"""
Mask Intersection Engine — Combines damage + anatomy predictions.

For each detected damage region, determines which car part(s) it
falls on, computes area ratios, and produces bounding boxes.

Includes false-positive suppression for:
  - Noise (tiny regions)
  - Edge artifacts (predictions at image borders)
  - Ultra-thin scratches that are likely noise
  - Reflection-shaped false dents
"""
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Optional

from pipeline.config import (
    DAMAGE_CLASSES, ANATOMY_CLASSES,
    FP_MIN_DAMAGE_AREA, FP_MAX_SCRATCH_ASPECT,
    FP_EDGE_MARGIN_PX,
)


# Reverse maps: id → name
DAMAGE_ID_TO_NAME  = {v: k for k, v in DAMAGE_CLASSES.items()}
ANATOMY_ID_TO_NAME = {v: k for k, v in ANATOMY_CLASSES.items()}


@dataclass
class DamageInstance:
    """A single detected damage region with part context."""
    damage_type: str                         # e.g. "dent"
    damage_class_id: int
    affected_parts: List[str] = field(default_factory=list)
    area_pixels: int = 0
    area_percentage: float = 0.0
    relative_area_percentage: float = 0.0    # relative to entire detected car area
    context_parts_count: int = 1             # scale/zoom estimation proxy
    bbox: List[int] = field(default_factory=list)  # [x1, y1, x2, y2]
    mask: Optional[np.ndarray] = None        # binary mask for this instance
    centroid: tuple = (0, 0)
    contour_complexity: float = 0.0          # perimeter²/area — edge jaggedness
    solidity: float = 0.0                    # convex_area / area — shape regularity
    is_edge_region: bool = False             # near image border


def _is_edge_region(bbox: List[int], H: int, W: int,
                    margin: int = FP_EDGE_MARGIN_PX) -> bool:
    """Check if a bounding box touches the image edge."""
    x1, y1, x2, y2 = bbox
    return (x1 <= margin or y1 <= margin or
            x2 >= W - margin or y2 >= H - margin)


def _compute_solidity(contours) -> float:
    """Solidity = contour area / convex hull area. Low = noisy shape."""
    if not contours:
        return 0.0
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)
    return area / max(hull_area, 1)


def _aspect_ratio(bbox: List[int]) -> float:
    """Width / height ratio of the bounding box."""
    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)
    return max(w / h, h / w)


def extract_damage_instances(damage_map: np.ndarray,
                             anatomy_map: np.ndarray,
                             quality_report=None) -> List[DamageInstance]:
    """
    Extract individual damage instances from semantic prediction maps,
    intersect with anatomy predictions, and filter false positives.

    Args:
        damage_map:     (H, W) int array, values in DAMAGE_CLASSES.
        anatomy_map:    (H, W) int array, values in ANATOMY_CLASSES.
        quality_report: Optional QualityReport for context-aware filtering.

    Returns:
        List of DamageInstance with part context.
    """
    H, W = damage_map.shape
    total_pixels_image = H * W
    
    # Calculate total visible vehicle area (any non-background anatomy)
    car_mask = (anatomy_map > 0).astype(np.uint8)
    total_car_pixels = int(car_mask.sum())
    
    # If no car detected, fall back to image size.
    # However, if it's a tight zoom, the entire image IS the car.
    # We shouldn't penalize relative area just because the model didn't see explicit boundaries.
    if total_car_pixels == 0:
        total_car_pixels = total_pixels_image

    # Context parts count (how much of the car is visible = zoom estimation proxy)
    unique_parts_detected = len([p for p in np.unique(anatomy_map) if p != 0])

    instances = []

    # Context flags from quality report
    has_reflections = False
    has_mud = False
    has_water = False
    if quality_report is not None:
        has_reflections = getattr(quality_report,
                                  'needs_reflection_suppression', False)
        has_mud = getattr(quality_report, 'has_mud_patches', False)
        has_water = getattr(quality_report, 'has_water_droplets', False)

    for cls_id in range(1, len(DAMAGE_CLASSES)):     # skip background
        cls_mask = (damage_map == cls_id).astype(np.uint8)
        if cls_mask.sum() == 0:
            continue

        damage_name = DAMAGE_ID_TO_NAME[cls_id]

        # ── Morphological Dilation ────────────────────────────────────────
        # Fuse scattered fragmented masks that belong to the same damage class.
        # Use an even larger kernel to aggressively merge nearby predictions
        # (e.g. the two pieces of the dent on the grey car).
        kernel_size = 15 if damage_name in ("scratch", "crack") else 101
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        cls_mask = cv2.dilate(cls_mask, kernel, iterations=1)
        # Erode back slightly to preserve general shape boundaries
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size//2, kernel_size//2))
        cls_mask = cv2.erode(cls_mask, erode_kernel, iterations=1)

        # Find connected components for this class
        num_labels, labels = cv2.connectedComponents(cls_mask)

        for label_id in range(1, num_labels):        # skip background label
            instance_mask = (labels == label_id).astype(np.uint8)
            area = int(instance_mask.sum())

            # ── FP Filter 1: Minimum area ─────────────────────────────────
            if area < FP_MIN_DAMAGE_AREA:
                continue

            # Bounding box
            ys, xs = np.where(instance_mask)
            x1, y1 = int(xs.min()), int(ys.min())
            x2, y2 = int(xs.max()), int(ys.max())
            bbox = [x1, y1, x2, y2]

            # ── FP Filter 2: Edge artifacts ───────────────────────────────
            edge = _is_edge_region(bbox, H, W)
            # If region is ONLY at the edge and small, likely an artifact
            if edge and area < FP_MIN_DAMAGE_AREA * 3:
                continue

            # Centroid
            cx = int(xs.mean())
            cy = int(ys.mean())

            # Contour analysis
            contours, _ = cv2.findContours(
                instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            perimeter = sum(cv2.arcLength(c, True) for c in contours)
            complexity = (perimeter ** 2) / max(area, 1)
            solidity = _compute_solidity(contours)
            aspect = _aspect_ratio(bbox)

            # ── FP Filter 3: Noise-like scratches (too thin) ─────────────
            if damage_name == "scratch" and aspect > FP_MAX_SCRATCH_ASPECT:
                continue  # impossibly thin line = noise

            # ── FP Filter 4: Reflection-shaped false dents ────────────────
            # Reflections base rules
            if has_reflections:
                if damage_name == "dent" and solidity > 0.85 and area < 3000:
                    # Very smooth, circular object on reflective surface = FP
                    continue
                if damage_name == "scratch" and area > 1000 and solidity > 0.6:
                    # Broad, sprawling "scratch" with solid shape = likely specular glare tracking along a panel
                    continue
                
            # Intersect with anatomy map
            anatomy_in_region = anatomy_map[instance_mask == 1]
            part_ids = np.unique(anatomy_in_region)
            affected_parts = [
                ANATOMY_ID_TO_NAME[pid]
                for pid in part_ids
                if pid in ANATOMY_ID_TO_NAME and pid != 0
            ]
            
            # ── FP Filter 5.5: Headlight Complex Reflections ────────────────
            if "lamp_mirror" in affected_parts:
                if damage_name in ("severe_break", "scratch", "crack"):
                    # Headlights have complex internal geometry (bulbs, LED strips, reflections)
                    # that models constantly mistake for scratches, cracks, or shattered plastic.
                    # Unless it's an incredibly jagged, complex shape (true shattered plastic),
                    # err on the side of caution and drop it.
                    if solidity > 0.55 or complexity < 40:
                        continue
                        
            # ── FP Filter 5: Mud confusion with scratches ────────────────
            if (has_mud and damage_name == "scratch" and
                    solidity > 0.7 and area < 1500):
                # Blob-shaped "scratch" on muddy car = likely dirt
                continue

            # ── FP Filter 6: Water droplet confusion ─────────────────────
            if (has_water and damage_name in ("scratch", "crack") and
                    area < 500):
                # Tiny damage prediction on wet surface = likely droplet
                continue

            # Intersect with anatomy map is now computed earlier for FP Filter 5.5

            instances.append(DamageInstance(
                damage_type=damage_name,
                damage_class_id=cls_id,
                affected_parts=affected_parts if affected_parts else ["unknown"],
                area_pixels=area,
                area_percentage=round(100.0 * area / total_pixels_image, 2),
                relative_area_percentage=round(100.0 * area / total_car_pixels, 2),
                context_parts_count=unique_parts_detected,
                bbox=bbox,
                mask=instance_mask,
                centroid=(cx, cy),
                contour_complexity=round(complexity, 2),
                solidity=round(solidity, 3),
                is_edge_region=edge,
            ))

    return instances
