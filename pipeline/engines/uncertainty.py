"""
MC Dropout Uncertainty — Epistemic uncertainty via Monte Carlo Dropout.

- Enables dropout in the Mask2Former decoder at inference time.
- Runs N forward passes and measures prediction variance.
- Routes predictions into: auto-approve / human-review / reject.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple

from pipeline.config import (
    MC_DROPOUT_PASSES, UNCERTAINTY_LOW, UNCERTAINTY_HIGH, IMG_SIZE,
)


def enable_mc_dropout(model):
    """
    Enable dropout layers in the model for MC inference.
    Only affects Dropout modules; BatchNorm stays in eval mode.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def disable_mc_dropout(model):
    """Restore eval mode for all dropout layers."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()


def mc_inference(backbone, decoder, image_tensor: torch.Tensor,
                 n_passes: int = MC_DROPOUT_PASSES) -> dict:
    """
    Run Monte Carlo Dropout inference.

    Args:
        backbone: DINOv2Backbone (stays frozen/eval).
        decoder:  DualMask2Former (dropout enabled).
        image_tensor: (1, 3, H, W) normalised tensor.
        n_passes: number of stochastic forward passes.

    Returns:
        dict with mean predictions, uncertainty maps, and routing decision.
    """
    device = image_tensor.device

    # Backbone forward (deterministic)
    with torch.no_grad():
        fpn_features = backbone(image_tensor)

    # Enable MC dropout in decoder
    enable_mc_dropout(decoder)

    damage_preds  = []
    anatomy_preds = []

    for _ in range(n_passes):
        with torch.no_grad():
            dmg_logits, anat_logits, _ = decoder(fpn_features)

        damage_preds.append(F.softmax(dmg_logits, dim=1))
        anatomy_preds.append(F.softmax(anat_logits, dim=1))

    # Disable MC dropout
    disable_mc_dropout(decoder)

    # Stack: (N, B, C, H, W)
    damage_stack  = torch.stack(damage_preds, dim=0)
    anatomy_stack = torch.stack(anatomy_preds, dim=0)

    # Mean prediction
    damage_mean  = damage_stack.mean(dim=0)     # (B, C, H, W)
    anatomy_mean = anatomy_stack.mean(dim=0)

    # Predictive uncertainty: variance of the softmax probabilities
    damage_var  = damage_stack.var(dim=0).mean(dim=1)   # (B, H, W)
    anatomy_var = anatomy_stack.var(dim=0).mean(dim=1)

    # Scalar uncertainty metrics
    damage_uncertainty  = float(damage_var.mean().item())
    anatomy_uncertainty = float(anatomy_var.mean().item())
    overall_uncertainty = (damage_uncertainty + anatomy_uncertainty) / 2

    # Routing decision
    if overall_uncertainty < UNCERTAINTY_LOW:
        routing = "auto_approve"
    elif overall_uncertainty > UNCERTAINTY_HIGH:
        routing = "reject_retake"
    else:
        routing = "human_review"

    return {
        "damage_probs":  damage_mean,
        "anatomy_probs": anatomy_mean,
        "damage_uncertainty_map":  damage_var,
        "anatomy_uncertainty_map": anatomy_var,
        "damage_uncertainty":  round(damage_uncertainty, 4),
        "anatomy_uncertainty": round(anatomy_uncertainty, 4),
        "overall_uncertainty": round(overall_uncertainty, 4),
        "routing": routing,
    }


def get_confidence_per_instance(uncertainty_map: np.ndarray,
                                instance_mask: np.ndarray) -> float:
    """
    Compute per-instance confidence from the uncertainty map.

    confidence = 1 - mean_uncertainty_in_mask_region
    """
    region_uncertainty = uncertainty_map[instance_mask == 1]
    if len(region_uncertainty) == 0:
        return 0.0
    mean_unc = float(np.mean(region_uncertainty))
    # Adjusted scaling factor: variance is usually small (e.g. 0.01 - 0.1).
    # Multiplying by 10 was too severe, causing confidence to drop significantly.
    # Multiplying by 2 or 3 provides a more reasonable penalty for uncertainty.
    return round(max(0.0, 1.0 - mean_unc * 3), 3)
