"""
End-to-End Inference Pipeline.

Full pipeline: VLM Context → Quality Gate → Enhancement → DINOv2 → Mask2Former
→ Mask Intersection → Missing Parts → Severity → Func/Cosmetic → MC Dropout
→ VLM Rationale → JSON + Overlay

Usage:
    python -m pipeline.inference --input path/to/images --output results/
    python -m pipeline.inference --input single_image.jpg --output results/
"""
import os
import sys
import glob
import time
import cv2
import torch
import numpy as np
import json

from pipeline.config import (
    BASE, CHECKPOINT_DIR, RESULTS_DIR, IMG_SIZE, IMG_MEAN, IMG_STD,
    IGNORE_INDEX, TRAIN_MANIFEST, VAL_MANIFEST,
    CONFIDENCE_THRESHOLD, MC_DROPOUT_PASSES,
    NUM_DAMAGE_CLASSES, NUM_ANATOMY_CLASSES,
    FP_LOW_CONFIDENCE_PRUNE, VLM_ENABLED,
)
from pipeline.quality_gate import run_quality_gate
from pipeline.enhancement import enhance_image
from pipeline.models.backbone import DINOv2Backbone
from pipeline.models.mask2former import DualMask2Former
from pipeline.models.severity_model import SeverityClassifier
from pipeline.engines.intersection import extract_damage_instances, ANATOMY_ID_TO_NAME
from pipeline.engines.missing_parts import detect_missing_parts
from pipeline.engines.severity import geometric_severity, classifier_severity, fuse_severity
from pipeline.engines.rules import classify_functional_cosmetic
from pipeline.engines.uncertainty import mc_inference, get_confidence_per_instance
from pipeline.engines.vlm_context import VLMContextEngine
from pipeline.engines.vlm_rationale import generate_rationale
from pipeline.engines.vlm_fusion import fuse_vlm_with_damages
from pipeline.engines.explainability import (
    build_json_output, format_damage_for_json,
    draw_overlay, draw_uncertainty_heatmap, save_results,
)


class DamagePipeline:
    """
    Full end-to-end vehicle damage detection pipeline.

    Load once, run on many images.
    """

    def __init__(self, checkpoint_dir: str = CHECKPOINT_DIR,
                 device: str = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Pipeline device: {self.device}")

        # ── Load Models ───────────────────────────────────────────────────
        print("Loading backbone (DINOv2)...")
        self.backbone = DINOv2Backbone(freeze=True).to(self.device)
        self.backbone.eval()

        print("Loading decoder (DualMask2Former)...")
        self.decoder = DualMask2Former().to(self.device)

        # Load trained weights
        seg_ckpt = os.path.join(checkpoint_dir, "best.pt")
        if os.path.exists(seg_ckpt):
            ckpt = torch.load(seg_ckpt, map_location=self.device)
            self.backbone.fpn.load_state_dict(ckpt["fpn_state"])
            self.decoder.load_state_dict(ckpt["decoder_state"])
            print(f"  ✓ Loaded segmentation weights (epoch {ckpt['epoch']})")
        else:
            print(f"  ⚠ No segmentation checkpoint at {seg_ckpt}")

        self.decoder.eval()

        print("Loading severity classifier...")
        self.severity_model = SeverityClassifier(pretrained=False).to(self.device)
        sev_ckpt = os.path.join(checkpoint_dir, "severity_best.pt")
        if os.path.exists(sev_ckpt):
            self.severity_model.load_state_dict(
                torch.load(sev_ckpt, map_location=self.device)
            )
            print("  ✓ Loaded severity weights")
        else:
            print(f"  ⚠ No severity checkpoint at {sev_ckpt}")

        self.severity_model.eval()

        # ── Load VLM Context Engine ───────────────────────────────────────
        self.vlm_engine = None
        if VLM_ENABLED:
            print("Loading VLM Context Engine (Qwen2.5-VL)...")
            try:
                self.vlm_engine = VLMContextEngine(device=str(self.device))
                if self.vlm_engine.available:
                    print("  ✓ VLM loaded")
                else:
                    print("  ⚠ VLM not available — falling back to heuristics")
            except Exception as e:
                print(f"  ⚠ VLM failed to load: {e}")
                self.vlm_engine = None

    def preprocess(self, image_bgr: np.ndarray) -> torch.Tensor:
        """BGR image → normalised tensor (1, 3, 512, 512)."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
        tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0

        # Normalize
        mean = torch.tensor(IMG_MEAN).view(3, 1, 1)
        std  = torch.tensor(IMG_STD).view(3, 1, 1)
        tensor = (tensor - mean) / std

        return tensor.unsqueeze(0).to(self.device)

    def run(self, image_path: str) -> dict:
        """
        Run the full pipeline on a single image.

        Args:
            image_path: path to the input image.

        Returns:
            dict with full structured output + overlay image.
        """
        t_start = time.time()

        # ── 1. Load Image ─────────────────────────────────────────────────
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            return {"error": f"Cannot read image: {image_path}"}

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # ── 0. VLM Pre-Analysis (scene context) ──────────────────────────
        vlm_ctx = None
        vlm_status = "disabled"  # always track VLM status for diagnostics

        if self.vlm_engine and self.vlm_engine.available:
            print("  Running VLM pre-analysis...")
            vlm_ctx = self.vlm_engine.analyze(image_path)
            if vlm_ctx and vlm_ctx.vlm_available:
                vlm_status = "success"
                print(f"  ✓ VLM: {vlm_ctx.vehicle_count} vehicle(s), "
                      f"type={vlm_ctx.vehicle_type_hint}, "
                      f"damage_visible={vlm_ctx.has_visible_damage}")
            else:
                vlm_status = "analysis_failed"
                print("  ⚠ VLM analysis returned no results")
        elif self.vlm_engine and not self.vlm_engine.available:
            vlm_status = "load_failed"
            print("  ⚠ VLM was enabled but failed to load")
        else:
            print("  ℹ VLM disabled or not configured")

        # ── 2. Quality Gate ───────────────────────────────────────────────
        quality_report = run_quality_gate(image_bgr)

        # Always include VLM status in quality report (even if VLM failed)
        quality_report.vlm_context = {"vlm_status": vlm_status}

        # Enrich quality report with VLM context if available
        if vlm_ctx and vlm_ctx.vlm_available:
            quality_report.vlm_context.update(vlm_ctx.to_dict())
            # Merge VLM surface conditions into warnings
            for cond in vlm_ctx.surface_conditions:
                if "mud" in cond and not quality_report.has_mud_patches:
                    quality_report.has_mud_patches = True
                    quality_report.warnings.append(
                        f"VLM detected mud/dirt on surface"
                    )
                if ("wet" in cond or "water" in cond) and not quality_report.has_water_droplets:
                    quality_report.has_water_droplets = True
                    quality_report.warnings.append(
                        f"VLM detected wet/water on surface"
                    )

        if quality_report.rejected:
            return {
                "image_path": image_path,
                "quality_assessment": quality_report.to_dict(),
                "rejected": True,
                "reason": "; ".join(quality_report.issues),
                "latency_ms": round((time.time() - t_start) * 1000),
            }

        # ── 3. Conditional Enhancement ────────────────────────────────────
        enhanced_bgr = enhance_image(image_bgr, quality_report)
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

        # ── 4. Model Inference (MC Dropout) ───────────────────────────────
        tensor = self.preprocess(enhanced_bgr)

        uncertainty_info = mc_inference(
            self.backbone, self.decoder, tensor,
            n_passes=MC_DROPOUT_PASSES
        )

        # Get argmax predictions
        damage_map = uncertainty_info["damage_probs"].argmax(dim=1)[0]
        damage_map = damage_map.cpu().numpy().astype(np.uint8)

        anatomy_map = uncertainty_info["anatomy_probs"].argmax(dim=1)[0]
        anatomy_map = anatomy_map.cpu().numpy().astype(np.uint8)

        # Resize maps back to original image size
        orig_h, orig_w = image_rgb.shape[:2]
        damage_map  = cv2.resize(damage_map,  (orig_w, orig_h),
                                  interpolation=cv2.INTER_NEAREST)
        anatomy_map = cv2.resize(anatomy_map, (orig_w, orig_h),
                                  interpolation=cv2.INTER_NEAREST)

        # Uncertainty map for per-instance confidence
        unc_map = uncertainty_info["damage_uncertainty_map"][0].cpu().numpy()
        unc_map = cv2.resize(unc_map, (orig_w, orig_h))

        # ── 5. Mask Intersection (quality-aware FP filtering) ─────────────
        damage_instances = extract_damage_instances(
            damage_map, anatomy_map, quality_report=quality_report
        )

        # ── 5b. Missing Parts Detection ───────────────────────────────────
        missing_parts = detect_missing_parts(anatomy_map, quality_report=quality_report)

        # ── 6. Per-instance analysis ──────────────────────────────────────
        damage_results = []
        for inst in damage_instances:
            # Severity
            geo = geometric_severity(inst)
            cls = classifier_severity(
                image_rgb, inst, self.severity_model, self.device
            )
            sev = fuse_severity(geo, cls)

            # Per-instance confidence
            conf = get_confidence_per_instance(
                unc_map, inst.mask
            )

            # Functional / cosmetic (quality-aware)
            func = classify_functional_cosmetic(
                inst,
                confidence=conf,
                quality_warnings=quality_report.warnings,
            )

            # Apply confidence modifier from rules engine
            adjusted_conf = conf * func.get("confidence_modifier", 1.0)

            # Skip low-confidence predictions (FP pruning)
            if adjusted_conf < FP_LOW_CONFIDENCE_PRUNE:
                continue

            result = format_damage_for_json(inst, sev, func, adjusted_conf)
            if func.get("disclaimers"):
                result["disclaimers"] = func["disclaimers"]
            damage_results.append(result)

        # ── 6a. VLM-Aware 2W Suppression ──────────────────────────────────
        #  Motorcycles have exposed engines, frames, forks, chains that the
        #  car-trained model mistakes for severe_break. Use VLM vehicle type
        #  to suppress ONLY these specific false positives.
        #  IMPORTANT: do NOT penalise other damage types (dent, scratch) —
        #  a dent on a motorcycle tank is still a real dent.
        is_2w = (vlm_ctx and vlm_ctx.vlm_available
                 and vlm_ctx.vehicle_type_hint == "2W")

        if is_2w and damage_results:
            filtered_results = []
            suppressed_count = 0
            has_exposed = getattr(vlm_ctx, 'exposed_mechanicals', False)

            for d in damage_results:
                # Only suppress: large severe_break with unknown parts on 2W
                #   = exposed engine/frame, not damage
                if (d.get("type") == "severe_break"
                        and d.get("area_percentage", 0) > 15
                        and d.get("affected_parts") == ["unknown"]):
                    reason = "exposed 2W engine/frame"
                    if has_exposed:
                        reason += " (VLM confirmed)"
                    print(f"  ⊘ Suppressed: severe_break ({d['area_percentage']:.1f}% area)"
                          f" — {reason}")
                    suppressed_count += 1
                    continue  # skip this damage entry

                # All other damage types pass through unchanged
                filtered_results.append(d)

            if suppressed_count > 0:
                print(f"  2W suppression: removed {suppressed_count} false positive(s)")

            damage_results = filtered_results

        # ── 6c. VLM-Segmenter Fusion ──────────────────────────────────────
        #  Cross-validate and enrich damage results with VLM context:
        #  fill unknown parts, severity cross-check, confidence adjustment.
        damage_results = fuse_vlm_with_damages(damage_results, vlm_ctx)

        # ── 6b. Append missing parts as damage entries ────────────────────
        for mp in missing_parts:
            damage_results.append({
                "type": "missing_part",
                "missing_part_name": mp.missing_part,
                "severity": "severe",
                "severity_score": 2,
                "confidence": mp.confidence,
                "classification": "functional",
                "classification_reason": mp.reason,
                "affected_parts": [mp.missing_part],
                "trigger_parts": mp.trigger_parts,
                "area_percentage": mp.gap_area_ratio * 100,
                "bbox": mp.approx_bbox,
                "centroid": [
                    (mp.approx_bbox[0] + mp.approx_bbox[2]) // 2,
                    (mp.approx_bbox[1] + mp.approx_bbox[3]) // 2,
                ] if mp.approx_bbox != [0, 0, 0, 0] else [0, 0],
                "contour_complexity": 0.0,
            })

        # ── 7. Detected Parts ─────────────────────────────────────────────
        unique_parts = np.unique(anatomy_map)
        parts_detected = [
            ANATOMY_ID_TO_NAME[pid]
            for pid in unique_parts
            if pid in ANATOMY_ID_TO_NAME and pid != 0
        ]

        # ── 7b. VLM False-Negative Safety Net ─────────────────────────────
        #  If VLM sees obvious damage but segmenter found nothing → manual review
        if vlm_ctx and vlm_ctx.override_no_damage and len(damage_results) == 0:
            uncertainty_info["routing"] = "human_review"
            uncertainty_info["vlm_override"] = True

        # ── 8. Build JSON Output ──────────────────────────────────────────
        json_output = build_json_output(
            image_path=image_path,
            damage_instances=damage_results,
            quality_report=quality_report.to_dict(),
            uncertainty_info=uncertainty_info,
            parts_detected=parts_detected,
        )

        # ── 8b. VLM Rationale Generation ──────────────────────────────────
        rationale = generate_rationale(self.vlm_engine, image_path, json_output)
        if rationale:
            json_output["rationale"] = rationale

        # ── 9. Generate Overlay & Heatmap ─────────────────────────────────
        overlay = draw_overlay(
            image_rgb, damage_instances, damage_results, anatomy_map,
            missing_parts=missing_parts,
        )
        
        heatmap = draw_uncertainty_heatmap(image_rgb, unc_map)

        latency = round((time.time() - t_start) * 1000)
        json_output["latency_ms"] = latency

        return {
            "json": json_output,
            "overlay": overlay,
            "heatmap": heatmap,
            "damage_map": damage_map,
            "anatomy_map": anatomy_map,
        }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Vehicle Damage Detection Pipeline"
    )
    parser.add_argument("--input", required=True,
                        help="Image path or directory of images")
    parser.add_argument("--output", default=RESULTS_DIR,
                        help="Output directory for results")
    parser.add_argument("--checkpoint", default=CHECKPOINT_DIR,
                        help="Checkpoint directory")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    pipeline = DamagePipeline(args.checkpoint, args.device)

    # Collect images
    if os.path.isdir(args.input):
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        image_paths = []
        for p in patterns:
            image_paths.extend(glob.glob(os.path.join(args.input, p)))
    else:
        image_paths = [args.input]

    print(f"\nProcessing {len(image_paths)} images...")

    for path in image_paths:
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(path)}")

        result = pipeline.run(path)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
            continue

        if result.get("rejected"):
            print(f"  REJECTED: {result['reason']}")
            continue

        json_output = result["json"]
        overlay = result["overlay"]
        heatmap = result.get("heatmap")

        # Save
        json_path, overlay_path, heatmap_path = save_results(
            args.output, os.path.basename(path),
            json_output, overlay, heatmap
        )

        # Summary
        n_damages = json_output["num_damages_detected"]
        severity  = json_output["overall_severity"]
        routing   = json_output["routing"]
        latency   = json_output["latency_ms"]

        print(f"  Damages: {n_damages} | Severity: {severity} | "
              f"Routing: {routing} | Latency: {latency}ms")

        for d in json_output["damages"]:
            print(f"    → {d['type']} on {', '.join(d['affected_parts'])} "
                  f"({d['severity']}, {d['classification']}, "
                  f"conf={d['confidence']:.0%})")

        print(f"  Saved: {json_path}")
        print(f"         {overlay_path}")
        if heatmap_path:
            print(f"         {heatmap_path}")

    print(f"\n{'='*60}")
    print("Done!")


if __name__ == "__main__":
    main()
