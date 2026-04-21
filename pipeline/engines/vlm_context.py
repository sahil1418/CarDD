"""
VLM Context Engine — Qwen2.5-VL scene understanding before segmentation.

Uses a Vision-Language Model to semantically understand the image BEFORE
the segmentation model runs. Detects scene-level context that OpenCV
heuristics cannot: multi-vehicle collisions, obvious damage descriptions,
lighting conditions, surface conditions.

Key capability: if VLM sees obvious damage but segmenter detects nothing,
routes to manual_review instead of auto_approve (false-negative safety net).
"""
import torch
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from pipeline.config import VLM_MODEL_NAME, VLM_ENABLED, VLM_CONTEXT_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class VLMContext:
    """Scene-level context from Vision-Language Model analysis."""
    # Vehicle identification
    vehicle_count: int = 1
    vehicle_type_hint: str = "4W"           # "2W", "3W", "4W", "unknown"
    vehicle_angle: str = "unknown"          # viewing angle

    # Damage assessment
    has_visible_damage: bool = False         # VLM thinks damage is present
    damage_count: int = 0                    # number of distinct damage areas
    damage_types: List[str] = field(default_factory=list)    # e.g. ["dent", "scratch"]
    damage_locations: List[str] = field(default_factory=list) # e.g. ["front_bumper"]
    damage_severity: str = "none"            # overall: minor/moderate/severe/none
    visible_damage_description: str = ""     # free-text (from DAMAGE_LOCATIONS + DAMAGE_TYPES)

    # Vehicle condition
    exposed_mechanicals: bool = False        # True for motorcycles with visible engine/chain
    paint_condition: str = "unknown"         # good/faded/chipped/peeling/discolored

    # Environment
    lighting: str = "unknown"
    surface_conditions: List[str] = field(default_factory=list)

    # Pipeline control
    confidence_modifiers: Dict[str, float] = field(default_factory=dict)
    override_no_damage: bool = False         # safety net for false negatives
    raw_response: str = ""                   # full VLM output for debugging
    vlm_available: bool = False

    def to_dict(self) -> dict:
        return {
            "vehicle_count": self.vehicle_count,
            "vehicle_type_hint": self.vehicle_type_hint,
            "vehicle_angle": self.vehicle_angle,
            "has_visible_damage": self.has_visible_damage,
            "damage_count": self.damage_count,
            "damage_types": self.damage_types,
            "damage_locations": self.damage_locations,
            "damage_severity": self.damage_severity,
            "visible_damage_description": self.visible_damage_description,
            "exposed_mechanicals": self.exposed_mechanicals,
            "paint_condition": self.paint_condition,
            "lighting": self.lighting,
            "surface_conditions": self.surface_conditions,
            "confidence_modifiers": self.confidence_modifiers,
            "override_no_damage": self.override_no_damage,
            "vlm_available": self.vlm_available,
        }


class VLMContextEngine:
    """
    Wraps Qwen2.5-VL for scene-level context extraction.

    Loads the model once, runs on every image before segmentation.
    Uses 4-bit AWQ quantization to keep VRAM usage ~6GB.
    """

    def __init__(self, model_name: str = VLM_MODEL_NAME, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.available = False

        if not VLM_ENABLED:
            logger.info("VLM Context Engine disabled in config")
            return

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

            print(f"  Loading VLM: {model_name}...")

            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
            )

            # Strategy: try 4-bit quantization first (saves VRAM ~6GB),
            # fall back to float16 (~14GB) if bitsandbytes is unavailable
            try:
                import bitsandbytes  # noqa: F401
                print("  Attempting 4-bit quantized loading...")
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    load_in_4bit=True,
                )
                print("  ✓ VLM loaded (4-bit quantized)")
            except Exception as e4bit:
                print(f"  ⚠ 4-bit loading failed ({e4bit}), trying float16...")
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
                print("  ✓ VLM loaded (float16)")

            self.available = True

        except ImportError as e:
            print(
                f"  ✗ VLM dependencies not installed ({e}). "
                "Install: pip install transformers qwen-vl-utils"
            )
        except Exception as e:
            print(f"  ✗ VLM failed to load: {e}")

    def analyze(self, image_path: str) -> VLMContext:
        """
        Run VLM scene analysis on a single image.

        Args:
            image_path: path to the input image.

        Returns:
            VLMContext with scene-level information.
        """
        ctx = VLMContext()

        if not self.available or self.model is None:
            return ctx

        try:
            from qwen_vl_utils import process_vision_info

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{image_path}"},
                        {"type": "text", "text": VLM_CONTEXT_PROMPT},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.1,
                    do_sample=False,
                )

            # Decode only the generated tokens (skip input tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            ctx.raw_response = response
            ctx.vlm_available = True

            # Parse the structured response
            ctx = self._parse_response(response, ctx)

        except Exception as e:
            logger.warning(f"VLM analysis failed: {e}")
            ctx.vlm_available = False

        return ctx

    def _parse_response(self, response: str, ctx: VLMContext) -> VLMContext:
        """
        Parse the VLM's structured text response into VLMContext fields.

        Handles 12 structured fields from the enhanced prompt.
        """
        lines = response.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line or ":" not in line:
                continue

            key, _, value = line.partition(":")
            key = key.strip().upper().replace(" ", "_")
            value = value.strip()

            if not value:
                continue

            if key == "VEHICLES":
                try:
                    ctx.vehicle_count = max(1, int(value.split()[0]))
                except (ValueError, IndexError):
                    ctx.vehicle_count = 1

            elif key == "VEHICLE_TYPE":
                vtype = value.upper().strip()
                if vtype in ("2W", "3W", "4W"):
                    ctx.vehicle_type_hint = vtype

            elif key == "VEHICLE_ANGLE":
                ctx.vehicle_angle = value.lower().strip()

            elif key == "DAMAGE_VISIBLE":
                ctx.has_visible_damage = value.upper().startswith("YES")

            elif key == "DAMAGE_COUNT":
                try:
                    ctx.damage_count = max(0, int(value.split()[0]))
                except (ValueError, IndexError):
                    ctx.damage_count = 0

            elif key == "DAMAGE_TYPES":
                ctx.damage_types = [
                    s.strip().lower()
                    for s in value.split(",")
                    if s.strip() and s.strip().lower() != "none"
                ]

            elif key == "DAMAGE_LOCATIONS":
                ctx.damage_locations = [
                    s.strip().lower()
                    for s in value.split(",")
                    if s.strip() and s.strip().lower() != "none"
                ]
                # Build a human-readable description from locations + types
                if ctx.damage_locations:
                    ctx.visible_damage_description = (
                        f"{', '.join(ctx.damage_types or ['damage'])} "
                        f"at {', '.join(ctx.damage_locations)}"
                    )

            elif key == "DAMAGE_SEVERITY":
                sev = value.lower().strip()
                if sev in ("minor", "moderate", "severe", "none"):
                    ctx.damage_severity = sev

            elif key == "EXPOSED_MECHANICALS":
                ctx.exposed_mechanicals = value.upper().startswith("YES")

            elif key == "PAINT_CONDITION":
                ctx.paint_condition = value.lower().strip()

            elif key == "LIGHTING":
                ctx.lighting = value.lower().strip()

            elif key == "SURFACE":
                ctx.surface_conditions = [
                    s.strip().lower()
                    for s in value.split(",")
                    if s.strip() and s.strip().lower() != "none"
                ]

        # ── Derived intelligence ──────────────────────────────────────────

        # Override flag: if VLM sees damage, prevent auto-approve of "no damage"
        # Note: VLM sometimes says DAMAGE_VISIBLE=YES but DAMAGE_COUNT=0
        # (e.g. zoomed-in shots where it's hard to count distinct areas)
        if ctx.has_visible_damage or ctx.damage_count > 0:
            ctx.override_no_damage = True

        # Multi-vehicle scenes are inherently harder for the segmenter
        if ctx.vehicle_count > 1:
            ctx.confidence_modifiers["multi_vehicle"] = 0.85

        # 2W with exposed mechanicals = strong signal for FP suppression
        if ctx.exposed_mechanicals and ctx.vehicle_type_hint == "2W":
            ctx.confidence_modifiers["exposed_engine_2w"] = 1.0

        # Surface condition modifiers
        for cond in ctx.surface_conditions:
            if "mud" in cond or "dirt" in cond or "rusty" in cond:
                ctx.confidence_modifiers["mud_risk"] = 0.8
            if "water" in cond or "wet" in cond:
                ctx.confidence_modifiers["water_risk"] = 0.85
            if "reflective" in cond or "glare" in cond:
                ctx.confidence_modifiers["reflection_risk"] = 0.8

        # Paint condition modifiers
        if ctx.paint_condition in ("chipped", "peeling"):
            ctx.confidence_modifiers["paint_damage_risk"] = 0.85

        return ctx

