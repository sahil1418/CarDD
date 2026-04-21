"""
Explainability Engine — Structured JSON output + visual overlay generation.

Produces:
  1. JSON with all damage instances, severity, confidence, parts, routing
  2. Visual overlay image with coloured masks, bounding boxes, and labels
"""
import json
import cv2
import numpy as np
from typing import List, Optional
from datetime import datetime

from pipeline.engines.intersection import DamageInstance, DAMAGE_ID_TO_NAME
from pipeline.engines.missing_parts import MissingPartInstance
from pipeline.config import DAMAGE_CLASSES, ANATOMY_CLASSES


# ── Colour palette for overlay ────────────────────────────────────────────────
DAMAGE_COLOURS = {
    "scratch":      (255, 200, 0),      # amber
    "dent":         (0, 150, 255),       # blue
    "crack":        (255, 80, 80),       # red
    "severe_break": (200, 0, 200),       # magenta
    "missing_part": (255, 128, 0),       # orange
}

ANATOMY_COLOURS = {
    "bumper":     (100, 255, 100),
    "door":       (100, 100, 255),
    "fender":     (255, 255, 100),
    "hood_trunk": (255, 150, 50),
    "glass":      (150, 255, 255),
    "lamp_mirror":(255, 100, 255),
    "wheel":      (200, 200, 200), 
}


def build_json_output(
    image_path: str,
    damage_instances: List[dict],
    quality_report: dict,
    uncertainty_info: dict,
    parts_detected: List[str],
) -> dict:
    """
    Build the final structured JSON output.

    Args:
        image_path:        path to the input image.
        damage_instances:  list of per-damage dicts (from pipeline).
        quality_report:    from QualityGate.to_dict().
        uncertainty_info:  from mc_inference().
        parts_detected:    list of anatomy part names found.

    Returns:
        Complete JSON-serialisable dict.
    """
    # Determine overall severity
    if damage_instances:
        max_sev = max(d.get("severity_score", 0) for d in damage_instances)
        severity_map = {0: "minor", 1: "moderate", 2: "severe"}
        overall_severity = severity_map.get(max_sev, "unknown")
    else:
        overall_severity = "none"

    output = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "image_path": image_path,
        "quality_assessment": quality_report,
        "num_damages_detected": len(damage_instances),
        "damages": damage_instances,
        "vehicle_parts_detected": parts_detected,
        "overall_severity": overall_severity,
        "uncertainty": {
            "damage":  uncertainty_info.get("damage_uncertainty", None),
            "anatomy": uncertainty_info.get("anatomy_uncertainty", None),
            "overall": uncertainty_info.get("overall_uncertainty", None),
        },
        "routing": uncertainty_info.get("routing", "unknown"),
        "requires_human_review": uncertainty_info.get("routing") == "human_review",
    }

    return output


def format_damage_for_json(
    instance: DamageInstance,
    severity_info: dict,
    func_cosmetic: dict,
    confidence: float,
) -> dict:
    """Format a single DamageInstance for JSON output."""
    return {
        "type": instance.damage_type,
        "severity": severity_info.get("severity", "unknown"),
        "severity_score": severity_info.get("severity_score", 0),
        "confidence": confidence,
        "classification": func_cosmetic.get("classification", "unknown"),
        "classification_reason": func_cosmetic.get("reason", ""),
        "affected_parts": instance.affected_parts,
        "context_parts_count": severity_info.get("geometric", {}).get("context_parts_count", 1),
        "area_percentage": instance.area_percentage,
        "bbox": instance.bbox,
        "centroid": list(instance.centroid),
        "contour_complexity": instance.contour_complexity,
    }


def draw_overlay(
    image: np.ndarray,
    damage_instances: List[DamageInstance],
    damage_results: List[dict],
    anatomy_map: Optional[np.ndarray] = None,
    missing_parts: Optional[List[MissingPartInstance]] = None,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Draw visual overlay with coloured damage masks, bounding boxes,
    severity labels, and optionally anatomy regions.

    Args:
        image:            RGB uint8 (H, W, 3).
        damage_instances: list of DamageInstance objects.
        damage_results:   per-instance dicts with severity/classification.
        anatomy_map:      (H, W) anatomy class IDs, or None.
        alpha:            overlay transparency.

    Returns:
        RGB annotated image.
    """
    overlay = image.copy()

    # Draw anatomy underlay (if available)
    if anatomy_map is not None:
        anat_overlay = np.zeros_like(image)
        for name, cls_id in ANATOMY_CLASSES.items():
            if cls_id == 0:
                continue
            mask = (anatomy_map == cls_id)
            if mask.any() and name in ANATOMY_COLOURS:
                anat_overlay[mask] = ANATOMY_COLOURS[name]
        # Lighter blend for anatomy
        overlay = cv2.addWeighted(overlay, 1.0, anat_overlay, 0.2, 0)

    # Draw damage masks
    for inst, result in zip(damage_instances, damage_results):
        if inst.mask is None:
            continue

        colour = DAMAGE_COLOURS.get(inst.damage_type, (255, 255, 255))

        # Semi-transparent mask
        mask_overlay = np.zeros_like(image)
        mask_overlay[inst.mask == 1] = colour
        overlay = cv2.addWeighted(overlay, 1.0, mask_overlay, alpha, 0)

        # Bounding box
        x1, y1, x2, y2 = inst.bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, 2)

        # Label
        severity = result.get("severity", "?")
        confidence = result.get("confidence", 0)
        classification = result.get("classification", "?")
        label = f"{inst.damage_type} | {severity} | {classification} ({confidence:.0%})"

        # Background for text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # If the bounding box is too close to the top of the image, push the label inside the box
        if y1 - th - 8 < 0:
            cv2.rectangle(overlay, (x1, y1), (x1 + tw + 4, y1 + th + 8), colour, -1)
            cv2.putText(overlay, label, (x1 + 2, y1 + th + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                        cv2.LINE_AA)
        else:
            cv2.rectangle(overlay, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
            cv2.putText(overlay, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                        cv2.LINE_AA)

    # Draw missing-part indicators (dashed orange boxes)
    if missing_parts:
        mp_colour = DAMAGE_COLOURS.get("missing_part", (255, 128, 0))
        for mp in missing_parts:
            if not mp.approx_bbox or mp.approx_bbox == [0, 0, 0, 0]:
                continue
            x1, y1, x2, y2 = mp.approx_bbox

            # Dashed rectangle (draw segments)
            dash_len = 10
            for edge_pts in [
                ((x1, y1), (x2, y1)),   # top
                ((x2, y1), (x2, y2)),   # right
                ((x2, y2), (x1, y2)),   # bottom
                ((x1, y2), (x1, y1)),   # left
            ]:
                pt1, pt2 = edge_pts
                dist = int(np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]))
                for i in range(0, dist, dash_len * 2):
                    t1 = i / max(dist, 1)
                    t2 = min((i + dash_len) / max(dist, 1), 1.0)
                    sp = (int(pt1[0] + t1 * (pt2[0] - pt1[0])),
                          int(pt1[1] + t1 * (pt2[1] - pt1[1])))
                    ep = (int(pt1[0] + t2 * (pt2[0] - pt1[0])),
                          int(pt1[1] + t2 * (pt2[1] - pt1[1])))
                    cv2.line(overlay, sp, ep, mp_colour, 2)

            # Label
            label = f"MISSING: {mp.missing_part} ({mp.confidence:.0%})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            if y1 - th - 8 < 0:
                cv2.rectangle(overlay, (x1, y1), (x1 + tw + 4, y1 + th + 8), mp_colour, -1)
                cv2.putText(overlay, label, (x1 + 2, y1 + th + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                cv2.rectangle(overlay, (x1, y1 - th - 8), (x1 + tw + 4, y1), mp_colour, -1)
                cv2.putText(overlay, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return overlay


def draw_uncertainty_heatmap(image: np.ndarray, uncertainty_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Convert a 2D float uncertainty/variance map into a colour heatmap overlaid on the original image.
    """
    if uncertainty_map.max() > 0:
        # Normalize to 0-255
        norm_map = (uncertainty_map / uncertainty_map.max() * 255).astype(np.uint8)
    else:
        norm_map = np.zeros_like(uncertainty_map, dtype=np.uint8)
        
    # Apply JET colormap (blue = low variance, red = high variance)
    heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
    
    # Blend with original image
    return cv2.addWeighted(image, 1.0 - alpha, heatmap, alpha, 0)

def save_results(output_dir: str, image_name: str,
                 json_output: dict, overlay_image: np.ndarray,
                 heatmap_image: Optional[np.ndarray] = None):
    """Save JSON and overlay image to disk."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(image_name)[0]

    json_path = os.path.join(output_dir, f"{base}_result.json")
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)

    overlay_path = os.path.join(output_dir, f"{base}_overlay.jpg")
    overlay_bgr = cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(overlay_path, overlay_bgr)
    
    heatmap_path = None
    if heatmap_image is not None:
        heatmap_path = os.path.join(output_dir, f"{base}_heatmap.jpg")
        heatmap_bgr = cv2.cvtColor(heatmap_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(heatmap_path, heatmap_bgr)

    return json_path, overlay_path, heatmap_path
