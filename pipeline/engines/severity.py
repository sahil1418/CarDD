"""
Physical Severity Engine — Two-stage severity estimation.

Stage 1: Geometric features from segmentation mask
  - Relative area, contour complexity, number of components

Stage 2: EfficientNet-B0 classifier on the cropped damage region

Fusion: weighted combination of both scores.
"""
import numpy as np
import cv2
import torch

from pipeline.config import SEVERITY_CLASSES
from pipeline.engines.intersection import DamageInstance


# Reverse map
SEVERITY_ID_TO_NAME = {v: k for k, v in SEVERITY_CLASSES.items()}


def geometric_severity(instance: DamageInstance) -> dict:
    """
    Estimate severity from mask geometry alone.

    Heuristics:
      - Uses context_parts_count to estimate zoom level.
      - If zoomed in (1-2 parts): Needs massive area to be severe.
      - If zoomed out (3+ parts): Small relative area is physically severe.
    """
    area = getattr(instance, 'relative_area_percentage', instance.area_percentage)
    complexity = instance.contour_complexity
    parts_count = getattr(instance, 'context_parts_count', 3)
    
    # Zoom-agnostic severity thresholds
    if parts_count <= 2:
        # Macro shot / Zoomed in
        severe_thresh, mod_thresh = 40.0, 15.0
    elif parts_count == 3:
        # Medium shot
        severe_thresh, mod_thresh = 15.0, 5.0
    else:
        # Zoomed out (4+ parts visible)
        severe_thresh, mod_thresh = 8.0, 3.0

    # Base severity from area
    if instance.damage_type == "severe_break":
        score = 2    # always severe
    elif area > severe_thresh or complexity > 80:
        score = 2    # severe
    elif area > mod_thresh or complexity > 40:
        score = 1    # moderate
    else:
        score = 0    # minor

    # Complexity bonus: jagged edges suggest worse damage
    if complexity > 60 and score < 2:
        score += 1

    return {
        "geometric_score": score,
        "geometric_label": SEVERITY_ID_TO_NAME[min(score, 2)],
        "area_pct": instance.area_percentage,
        "relative_area_pct": getattr(instance, 'relative_area_percentage', instance.area_percentage),
        "context_parts_count": getattr(instance, 'context_parts_count', 1),
        "complexity": instance.contour_complexity,
    }


def classifier_severity(image: np.ndarray, instance: DamageInstance,
                        severity_model, device: torch.device) -> dict:
    """
    Run the EfficientNet-B0 severity classifier on the cropped damage region.

    Args:
        image:          RGB uint8 (H, W, 3).
        instance:       DamageInstance with bbox.
        severity_model: loaded SeverityClassifier.
        device:         torch device.

    Returns:
        dict with classifier_score, classifier_label, classifier_conf.
    """
    x1, y1, x2, y2 = instance.bbox

    # Pad crop by 10% for context
    h, w = image.shape[:2]
    pad_x = max(int((x2 - x1) * 0.1), 5)
    pad_y = max(int((y2 - y1) * 0.1), 5)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return {"classifier_score": 0, "classifier_label": "minor",
                "classifier_conf": 0.0}

    crop = cv2.resize(crop, (224, 224))
    tensor = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)

    # Normalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    tensor = (tensor - mean) / std

    cls_idx, conf = severity_model.predict(tensor)
    cls_idx = cls_idx.item()
    conf = conf.item()

    return {
        "classifier_score": cls_idx,
        "classifier_label": SEVERITY_ID_TO_NAME[cls_idx],
        "classifier_conf": round(conf, 3),
    }


def fuse_severity(geometric: dict, classifier: dict,
                  geo_weight: float = 0.4,
                  cls_weight: float = 0.6) -> dict:
    """
    Weighted fusion of geometric and classifier severity.

    If classifier confidence is low (< 0.5), trust geometry more.
    """
    cls_conf = classifier.get("classifier_conf", 0.0)

    # Adjust weights if classifier is uncertain
    if cls_conf < 0.5:
        gw, cw = 0.7, 0.3
    else:
        gw, cw = geo_weight, cls_weight

    # Override: if geometric rules indicate this is a massive damage relative to the car size,
    # trust the geometry. The classifier (which sees a zoomed crop) loses scale context.
    # Note: we use >= 2 context parts as a heuristic that the area is trustworthy for override
    if geometric.get("geometric_score") == 2 and geometric.get("relative_area_pct", 0) > 15.0 and geometric.get("context_parts_count", 0) >= 2:
        fused_idx = 2
        gw, cw = 1.0, 0.0
    else:
        fused_score = gw * geometric["geometric_score"] + cw * classifier["classifier_score"]
        fused_idx = int(round(fused_score))
        fused_idx = min(max(fused_idx, 0), 2)

    return {
        "severity": SEVERITY_ID_TO_NAME[fused_idx],
        "severity_score": fused_idx,
        "geometric": geometric,
        "classifier": classifier,
        "fusion_weights": {"geometric": gw, "classifier": cw},
    }
