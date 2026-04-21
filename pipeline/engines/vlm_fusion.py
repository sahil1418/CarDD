"""
VLM-Segmenter Fusion Engine — Cross-validate and enrich segmenter outputs.

After the segmentation model produces damage instances, this engine uses
the VLM's scene-level context to:
  1. Fill in unknown affected_parts from VLM damage_locations
  2. Cross-check severity between VLM and segmenter
  3. Adjust confidence based on agreement/disagreement
  4. Add VLM-informed disclaimers
"""
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# ── Severity mapping for comparison ──────────────────────────────────────────
_SEVERITY_RANK = {"none": -1, "minor": 0, "moderate": 1, "severe": 2}


def fuse_vlm_with_damages(
    damage_results: List[dict],
    vlm_ctx,
) -> List[dict]:
    """
    Enrich and cross-validate segmenter damage results using VLM context.

    Args:
        damage_results: list of damage dicts from the per-instance analysis.
        vlm_ctx:        VLMContext dataclass (or None if VLM unavailable).

    Returns:
        Updated damage_results with VLM enrichments applied.
    """
    if vlm_ctx is None or not vlm_ctx.vlm_available:
        return damage_results

    if not damage_results:
        return damage_results

    vlm_locations = vlm_ctx.damage_locations    # e.g. ["front_bumper", "headlight"]
    vlm_types = vlm_ctx.damage_types            # e.g. ["dent", "crack"]
    vlm_severity = vlm_ctx.damage_severity      # e.g. "severe"
    vlm_sees_damage = vlm_ctx.has_visible_damage

    for d in damage_results:
        # Skip missing_part entries — VLM fusion is for segmenter outputs
        if d.get("type") == "missing_part":
            continue

        # ── 1. Fill unknown parts from VLM damage_locations ──────────────
        if d.get("affected_parts") == ["unknown"] and vlm_locations:
            d["affected_parts"] = vlm_locations
            d["affected_parts_source"] = "vlm"
            print(f"  VLM→ Filled unknown parts: {vlm_locations}")

        # ── 2. Severity cross-check ──────────────────────────────────────
        seg_severity = d.get("severity", "minor")
        seg_rank = _SEVERITY_RANK.get(seg_severity, 0)
        vlm_rank = _SEVERITY_RANK.get(vlm_severity, -1)

        if vlm_rank >= 0 and seg_rank >= 0:
            diff = abs(vlm_rank - seg_rank)

            if diff == 0:
                # Agreement — boost confidence slightly
                d["confidence"] = min(d.get("confidence", 0.5) * 1.05, 1.0)
                d.setdefault("vlm_notes", []).append(
                    f"VLM severity agrees: {vlm_severity}"
                )
            elif diff == 1:
                # Minor disagreement — note it but don't change
                d.setdefault("vlm_notes", []).append(
                    f"VLM severity differs slightly: VLM={vlm_severity} vs seg={seg_severity}"
                )
            else:
                # Major disagreement (2+ ranks apart) — add strong disclaimer
                d.setdefault("disclaimers", []).append(
                    f"Severity mismatch: AI model says {seg_severity} "
                    f"but VLM assessment says {vlm_severity} — manual review advised"
                )
                d.setdefault("vlm_notes", []).append(
                    f"VLM severity strongly disagrees: VLM={vlm_severity} vs seg={seg_severity}"
                )

        # ── 3. Confidence adjustment based on VLM agreement ──────────────
        seg_type = d.get("type", "")

        # If VLM's damage types include what the segmenter found → boost
        if vlm_types and seg_type in vlm_types:
            d["confidence"] = min(d.get("confidence", 0.5) * 1.10, 1.0)
            d.setdefault("vlm_notes", []).append(
                f"VLM confirms damage type: {seg_type}"
            )

        # If VLM says no damage at all but segmenter found some → penalise
        # Skip this for 2W: VLM often misses subtle motorcycle damage (tank dents)
        is_2w = getattr(vlm_ctx, 'vehicle_type_hint', '4W') == '2W'
        if not vlm_sees_damage and vlm_severity == "none" and not is_2w:
            d["confidence"] = d.get("confidence", 0.5) * 0.75
            d.setdefault("disclaimers", []).append(
                "VLM detected no visible damage — prediction may be a false positive"
            )

        # ── 4. Add VLM angle context ─────────────────────────────────────
        angle = vlm_ctx.vehicle_angle
        if angle and angle != "unknown":
            d["vehicle_angle"] = angle

    return damage_results
