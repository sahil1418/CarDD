"""
Functional / Cosmetic Rule Engine — Deterministic part-aware logic.

Rules:
  - Damage on glass, lamps, mirrors, wheels → FUNCTIONAL
  - severe_break on ANY part               → FUNCTIONAL
  - Surface damage (scratch/dent) on body panels → COSMETIC
  - crack on structural parts              → FUNCTIONAL

Edge-case handling:
  - Confidence-weighted classification
  - Quality-aware severity escalation (mud / water → uncertainty disclaimer)
  - Edge-region predictions flagged as lower confidence
"""
from pipeline.config import (
    FUNCTIONAL_PARTS, ALWAYS_FUNCTIONAL, FP_LOW_CONFIDENCE_PRUNE,
)
from pipeline.engines.intersection import DamageInstance


def classify_functional_cosmetic(instance: DamageInstance,
                                 confidence: float = 1.0,
                                 quality_warnings: list = None) -> dict:
    """
    Determines whether a damage instance is functional or cosmetic.

    Args:
        instance:          DamageInstance with damage_type and affected_parts.
        confidence:        Model confidence for this prediction (0-1).
        quality_warnings:  List of quality warning strings (from quality gate).

    Returns:
        dict with classification, reasoning, confidence, and disclaimers.
    """
    damage_type = instance.damage_type
    parts = set(instance.affected_parts)
    warnings = quality_warnings or []

    result = {
        "classification": "cosmetic",       # default
        "reason": "",
        "confidence_modifier": 1.0,
        "disclaimers": [],
    }

    # ── Confidence penalty for edge-region predictions ────────────────────
    if instance.is_edge_region:
        result["confidence_modifier"] *= 0.85
        result["disclaimers"].append(
            "Prediction at image edge — lower confidence"
        )

    # ── Quality-based disclaimers ─────────────────────────────────────────
    for w in warnings:
        if "mud" in w.lower() or "dirt" in w.lower():
            if damage_type == "scratch":
                result["confidence_modifier"] *= 0.8
                result["disclaimers"].append(
                    "Possible mud/dirt confusion — manual verification recommended"
                )

        if "water" in w.lower() or "droplet" in w.lower():
            if damage_type in ("scratch", "crack"):
                result["confidence_modifier"] *= 0.85
                result["disclaimers"].append(
                    "Water droplets present — may affect surface prediction"
                )

        if "reflection" in w.lower() or "specular" in w.lower():
            if damage_type == "dent":
                result["confidence_modifier"] *= 0.8
                result["disclaimers"].append(
                    "Specular reflections detected — dent may be reflection artifact"
                )

        if "matte" in w.lower():
            if damage_type == "dent":
                result["confidence_modifier"] *= 1.1  # matte = fewer reflections
                # Dent on matte is MORE reliable
                result["disclaimers"].append(
                    "Matte paint — dent prediction more reliable"
                )

    # ── Rule 1: severe_break on ANY part is always functional ─────────────
    if damage_type in ALWAYS_FUNCTIONAL:
        result["classification"] = "functional"
        result["reason"] = (
            f"{damage_type} damage always indicates "
            f"structural/functional impact"
        )
        return result

    # ── Rule 2: damage on inherently functional parts ─────────────────────
    functional_overlap = parts & FUNCTIONAL_PARTS
    if functional_overlap:
        result["classification"] = "functional"
        result["reason"] = (
            f"Damage affects functional component(s): "
            f"{', '.join(functional_overlap)}"
        )
        return result

    # ── Rule 3: crack on structural parts ─────────────────────────────────
    if damage_type == "crack" and parts - {"unknown"}:
        structural_parts = parts & {"fender", "hood_trunk", "bumper"}
        if structural_parts:
            result["classification"] = "functional"
            result["reason"] = (
                f"Crack on structural part(s): "
                f"{', '.join(structural_parts)} may compromise safety"
            )
            return result

    # ── Rule 4: large dent on structural parts may be functional ──────────
    if damage_type == "dent" and instance.area_percentage > 2.0:
        structural = parts & {"fender", "hood_trunk", "bumper", "door"}
        if structural:
            result["classification"] = "functional"
            result["reason"] = (
                f"Large dent ({instance.area_percentage:.1f}% area) "
                f"on {', '.join(structural)} — structural concern"
            )
            return result

    # ── Rule 5: surface damage on body panels → cosmetic ──────────────────
    result["classification"] = "cosmetic"
    result["reason"] = (
        f"Surface {damage_type} on body panel(s): {', '.join(parts)}"
    )

    return result
