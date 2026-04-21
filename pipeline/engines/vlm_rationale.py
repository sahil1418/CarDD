"""
VLM Rationale Generator — Natural language damage assessment explanations.

After the pipeline produces structured JSON results, feeds the image + results
back into Qwen2.5-VL to generate a human-readable rationale for claim adjusters.

Shares the same VLM model instance as vlm_context.py (passed in at init).
"""
import torch
import logging
from typing import Optional, Dict, Any

from pipeline.config import VLM_ENABLED, VLM_RATIONALE_PROMPT

logger = logging.getLogger(__name__)


def generate_rationale(
    vlm_engine,
    image_path: str,
    json_results: dict,
) -> str:
    """
    Generate a natural language rationale for the damage assessment.

    Args:
        vlm_engine:   VLMContextEngine instance (reused for model sharing).
        image_path:   path to the input image.
        json_results: the pipeline's structured JSON output.

    Returns:
        Natural language rationale string, or empty string if VLM unavailable.
    """
    if not VLM_ENABLED or vlm_engine is None or not vlm_engine.available:
        return ""

    try:
        from qwen_vl_utils import process_vision_info

        # Build a summary of the JSON results for the prompt
        damages_summary = _summarize_damages(json_results)
        prompt = VLM_RATIONALE_PROMPT.format(damages_summary=damages_summary)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = vlm_engine.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = vlm_engine.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(vlm_engine.device)

        with torch.no_grad():
            generated_ids = vlm_engine.model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.3,
                do_sample=True,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        rationale = vlm_engine.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return rationale

    except Exception as e:
        logger.warning(f"VLM rationale generation failed: {e}")
        return ""


def _summarize_damages(json_results: dict) -> str:
    """Convert JSON damage results into a concise text summary for the prompt."""
    damages = json_results.get("damages", [])
    parts = json_results.get("vehicle_parts_detected", [])
    severity = json_results.get("overall_severity", "none")

    if not damages:
        return (
            f"No damages detected by the AI model. "
            f"Vehicle parts visible: {', '.join(parts) if parts else 'none'}. "
            f"Overall severity: {severity}."
        )

    lines = [f"Overall severity: {severity}. {len(damages)} damage(s) found:"]
    for i, d in enumerate(damages, 1):
        dtype = d.get("type", "unknown")
        sev = d.get("severity", "unknown")
        conf = d.get("confidence", 0)
        cls = d.get("classification", "unknown")
        affected = ", ".join(d.get("affected_parts", ["unknown"]))
        area = d.get("area_percentage", 0)
        lines.append(
            f"  {i}. {dtype} on {affected} — {sev}, {cls}, "
            f"confidence={conf:.0%}, area={area:.1f}%"
        )

    lines.append(f"Visible parts: {', '.join(parts) if parts else 'none'}")
    return "\n".join(lines)
