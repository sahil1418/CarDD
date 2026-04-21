"""
Missing Parts Detection Engine — Detects vehicle parts that are expected but absent.

Uses the anatomy head predictions to infer which parts SHOULD be visible
based on co-occurrence rules (e.g., if headlights + hood are visible,
the bumper should also be visible). Flags missing parts as functional damage.

No retraining required — pure rule-engine logic on existing anatomy maps.
"""
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from pipeline.config import (
    ANATOMY_CLASSES,
    MISSING_PART_MIN_CAR_COVERAGE,
    MISSING_PART_MIN_GAP_AREA,
)


# Reverse map: id → name
ANATOMY_ID_TO_NAME = {v: k for k, v in ANATOMY_CLASSES.items()}


@dataclass
class MissingPartInstance:
    """A vehicle part that is expected but not detected."""
    missing_part: str                            # e.g. "bumper"
    trigger_parts: List[str] = field(default_factory=list)  # parts that imply it
    confidence: float = 0.0                      # how confident we are it's actually missing
    reason: str = ""                             # human-readable explanation
    approx_bbox: List[int] = field(default_factory=list)    # [x1, y1, x2, y2] estimated region
    gap_area_ratio: float = 0.0                  # void area / car area


# ── Co-occurrence rules ───────────────────────────────────────────────────────
# Each rule: (trigger_parts, expected_part, reason_template)
# Trigger parts: ALL must be present for the rule to fire.
# Expected part: the part that SHOULD be visible if triggers are present.
COOCCURRENCE_RULES = [
    (
        {"lamp_mirror", "hood_trunk"},
        "bumper",
        "Headlights/taillights and hood/trunk are visible, but bumper is absent — "
        "bumper may be missing or torn off"
    ),
    (
        {"door", "fender"},
        "wheel",
        "Door and fender are visible, but wheel is absent — "
        "wheel may be missing or obscured by damage"
    ),
    (
        {"hood_trunk", "bumper"},
        "glass",
        "Hood/trunk and bumper are visible, but windshield/glass is absent — "
        "glass may be shattered or missing"
    ),
    (
        {"lamp_mirror", "bumper"},
        "fender",
        "Headlights and bumper are visible, but fender is absent — "
        "fender may be missing or torn off"
    ),
]


def _estimate_gap_bbox(anatomy_map: np.ndarray,
                       trigger_part_ids: Set[int],
                       H: int, W: int) -> Tuple[List[int], float]:
    """
    Estimate approximate bounding box of the "gap" region where the
    missing part should be. Looks at the void area between/adjacent to
    trigger parts.

    Returns:
        (bbox [x1, y1, x2, y2], gap_area_ratio relative to car area)
    """
    # Build mask of trigger parts
    trigger_mask = np.zeros((H, W), dtype=np.uint8)
    for pid in trigger_part_ids:
        trigger_mask |= (anatomy_map == pid).astype(np.uint8)

    if trigger_mask.sum() == 0:
        return [0, 0, 0, 0], 0.0

    # Car mask (any non-background anatomy)
    car_mask = (anatomy_map > 0).astype(np.uint8)
    car_area = int(car_mask.sum())
    if car_area == 0:
        return [0, 0, 0, 0], 0.0

    # Dilate trigger parts to find the "expected neighbourhood"
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    expanded = cv2.dilate(trigger_mask, kernel, iterations=2)

    # Gap = expanded neighbourhood that is NOT any detected part (background void)
    gap_mask = expanded & (~car_mask.astype(bool)).astype(np.uint8)
    gap_area = int(gap_mask.sum())

    if gap_area == 0:
        return [0, 0, 0, 0], 0.0

    gap_ratio = gap_area / max(car_area, 1)

    # Bounding box of the gap
    ys, xs = np.where(gap_mask)
    if len(xs) == 0:
        return [0, 0, 0, 0], 0.0

    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())

    return [x1, y1, x2, y2], round(gap_ratio, 4)


def detect_missing_parts(anatomy_map: np.ndarray,
                         quality_report=None) -> List[MissingPartInstance]:
    """
    Detect vehicle parts that are expected but absent.

    Args:
        anatomy_map:    (H, W) int array, values in ANATOMY_CLASSES.
        quality_report: Optional QualityReport for context.

    Returns:
        List of MissingPartInstance objects.
    """
    H, W = anatomy_map.shape
    total_pixels = H * W

    # ── Safety guard 1: car must cover enough of the image ────────────────
    car_mask = (anatomy_map > 0).astype(np.uint8)
    car_pixels = int(car_mask.sum())
    car_coverage = car_pixels / max(total_pixels, 1)

    if car_coverage < MISSING_PART_MIN_CAR_COVERAGE:
        # Car too small/absent in frame — can't reliably judge missing parts
        return []

    # Detected parts (set of names)
    unique_ids = set(np.unique(anatomy_map))
    detected_parts: Set[str] = {
        ANATOMY_ID_TO_NAME[pid]
        for pid in unique_ids
        if pid in ANATOMY_ID_TO_NAME and pid != 0
    }

    # ── Safety guard 2: need at least 2 detected parts ────────────────────
    if len(detected_parts) < 2:
        return []

    missing_instances: List[MissingPartInstance] = []
    already_flagged: Set[str] = set()

    for trigger_parts, expected_part, reason in COOCCURRENCE_RULES:
        # Skip if expected part IS detected
        if expected_part in detected_parts:
            continue

        # Skip if already flagged by another rule
        if expected_part in already_flagged:
            continue

        # Check all trigger parts are present
        if not trigger_parts.issubset(detected_parts):
            continue

        # ── Spatial validation: check gap region ──────────────────────────
        trigger_ids = {
            ANATOMY_CLASSES[p] for p in trigger_parts
            if p in ANATOMY_CLASSES
        }
        bbox, gap_ratio = _estimate_gap_bbox(anatomy_map, trigger_ids, H, W)

        if gap_ratio < MISSING_PART_MIN_GAP_AREA:
            # Gap too small — part might just be out of frame
            continue

        # Confidence: higher gap ratio + more trigger parts = more confident
        confidence = min(0.95, 0.5 + gap_ratio * 2 + len(trigger_parts) * 0.05)

        missing_instances.append(MissingPartInstance(
            missing_part=expected_part,
            trigger_parts=sorted(trigger_parts),
            confidence=round(confidence, 3),
            reason=reason,
            approx_bbox=bbox,
            gap_area_ratio=gap_ratio,
        ))

        already_flagged.add(expected_part)

    return missing_instances
