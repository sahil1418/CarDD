"""
Pipeline Configuration — All hyperparams, paths, and class mappings.
"""
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = "/teamspace/studios/this_studio"
TRAIN_MANIFEST = os.path.join(BASE, "train_split.json")
VAL_MANIFEST   = os.path.join(BASE, "val_split.json")
CHECKPOINT_DIR = os.path.join(BASE, "checkpoints")
RESULTS_DIR    = os.path.join(BASE, "results")

# ─── Image ────────────────────────────────────────────────────────────────────
IMG_SIZE       = 518                     # must be divisible by patch_size=14 (518 = 14 × 37)
IMG_MEAN       = [0.485, 0.456, 0.406]
IMG_STD        = [0.229, 0.224, 0.225]
MIN_RESOLUTION = 224

# ─── DINOv2 Backbone ─────────────────────────────────────────────────────────
DINO_MODEL     = "dinov2_vitb14"
DINO_EMBED_DIM = 768
DINO_PATCH     = 14
DINO_LAYERS    = [2, 5, 8, 11]          # 0-indexed ViT-B block indices (12 blocks: 0-11)
FPN_CHANNELS   = 256
FREEZE_BACKBONE = True

# ─── Mask2Former Decoder ─────────────────────────────────────────────────────
NUM_QUERIES        = 100
DECODER_LAYERS     = 6
DECODER_DIM        = 256
DECODER_HEADS      = 8
DECODER_DROPOUT    = 0.1                # also used for MC Dropout at inference

# ─── Damage Head (A) ─────────────────────────────────────────────────────────
DAMAGE_CLASSES = {
    "background":   0,
    "scratch":      1,
    "dent":         2,
    "crack":        3,
    "severe_break": 4,
}
NUM_DAMAGE_CLASSES = len(DAMAGE_CLASSES)  # 5

# Class weights compensating for 3.93:1 imbalance (inverse-frequency based)
DAMAGE_CLASS_WEIGHTS = [0.5, 1.0, 2.2, 2.8, 3.9]

DAMAGE_REMAP = {
    "scratch": "scratch", "Scratch_or_spot": "scratch", "tray_son": "scratch",
    "damage": "scratch",
    "dent": "dent", "Dent": "dent", "Large_dent": "dent", "mop_lom": "dent",
    "crack": "crack", "Tear": "crack", "rach": "crack",
    "glass shatter": "severe_break", "lamp broken": "severe_break",
    "Large_tear_or_damage": "severe_break", "Shatter": "severe_break",
    "Dislocation": "severe_break", "mat_bo_phan": "severe_break",
}

# ─── Anatomy Head (B) ────────────────────────────────────────────────────────
ANATOMY_CLASSES = {
    "background": 0,
    "bumper":     1,
    "door":       2,
    "fender":     3,
    "hood_trunk": 4,
    "glass":      5,
    "lamp_mirror":6,
    "wheel":      7,
}
NUM_ANATOMY_CLASSES = len(ANATOMY_CLASSES)  # 8

ANATOMY_REMAP = {
    "Front-bumper": "bumper", "Back-bumper": "bumper", "Grille": "bumper",
    "Front-door": "door", "Back-door": "door",
    "Fender": "fender", "Quarter-panel": "fender", "Rocker-panel": "fender",
    "Hood": "hood_trunk", "Trunk": "hood_trunk",
    "Windshield": "glass", "Back-windshield": "glass",
    "Front-window": "glass", "Back-window": "glass",
    "Headlight": "lamp_mirror", "Tail-light": "lamp_mirror", "Mirror": "lamp_mirror",
    "Front-wheel": "wheel", "Back-wheel": "wheel",
}

# ─── Severity Classifier (C) ─────────────────────────────────────────────────
SEVERITY_CLASSES  = {"minor": 0, "moderate": 1, "severe": 2}
NUM_SEVERITY      = 3
SEVERITY_DATA_DIR = os.path.join(BASE, "severity_dataset", "data3a")

# ─── Training — Stage 1 (Dual-Head Segmentation) ─────────────────────────────
STAGE1_EPOCHS        = 12
STAGE1_BATCH_SIZE    = 8
STAGE1_GRAD_ACCUM    = 2                 # effective batch = 16
STAGE1_LR_DECODER    = 1e-4
STAGE1_LR_FPN        = 5e-5
STAGE1_LR_BACKBONE   = 1e-5             # only if LoRA is enabled
STAGE1_WEIGHT_DECAY  = 0.01
STAGE1_WARMUP_EPOCHS = 5
STAGE1_LOSS_CE_W     = 0.5
STAGE1_LOSS_DICE_W   = 0.5

# ─── Training — Stage 2 (Severity) ───────────────────────────────────────────
STAGE2_EPOCHS     = 5
STAGE2_BATCH_SIZE = 32
STAGE2_LR         = 1e-3

# ─── Inference ────────────────────────────────────────────────────────────────
MC_DROPOUT_PASSES    = 5
CONFIDENCE_THRESHOLD = 0.5
UNCERTAINTY_LOW      = 0.05              # auto-approve
UNCERTAINTY_HIGH     = 0.20              # request retake
IGNORE_INDEX         = 255

# ─── Quality Gate Thresholds ──────────────────────────────────────────────────
BLUR_THRESHOLD       = 80.0              # Laplacian variance
BLUR_SEVERE          = 30.0              # below this → reject (severe motion blur)
BRIGHTNESS_LOW       = 40                # mean pixel value
BRIGHTNESS_HIGH      = 220
MIN_IMAGE_DIM        = 224
WARN_IMAGE_DIM       = 384               # 224-384 → warn, lower confidence

# Glare / Hotspot Detection
GLARE_HOTSPOT_THRESH = 240               # pixel intensity considered "hot"
GLARE_HOTSPOT_RATIO  = 0.15              # if >15% pixels are hotspots → glare
GLARE_SATURATION_LOW = 20                # low saturation in hotspot = reflection

# Local Contrast (for reflection vs. dent differentiation)
LOCAL_CONTRAST_BLOCK = 32                # block size for local std computation
REFLECTION_CONTRAST_THRESH = 120         # very high local contrast = reflections

# Saturation Thresholds (mud / dirt / colour extremes)
SATURATION_LOW_THRESH  = 25              # very desaturated = potential grey/mud
SATURATION_HIGH_THRESH = 200             # oversaturated = colour aberration

# Colour Extremes (black/white car detection)
COLOR_EXTREME_DARK   = 50               # mean V < 50 = very dark car
COLOR_EXTREME_BRIGHT = 210              # mean V > 210 = very bright/white car

# ─── False Positive Suppression ───────────────────────────────────────────────
FP_MIN_DAMAGE_AREA      = 100           # minimum pixels for a valid damage region
FP_MAX_SCRATCH_ASPECT   = 25.0          # scratch aspect ratio cap (too thin = noise)
FP_EDGE_MARGIN_PX       = 8             # ignore predictions within 8px of image edge
FP_LOW_CONFIDENCE_PRUNE = 0.3           # remove predictions below 30% confidence

# ─── Missing Part Detection ──────────────────────────────────────────────────
MISSING_PART_MIN_CAR_COVERAGE = 0.15    # car must cover ≥15% of image
MISSING_PART_MIN_GAP_AREA     = 0.05    # void must be ≥5% of car area

# ─── Functional / Cosmetic Rules ──────────────────────────────────────────────
FUNCTIONAL_PARTS    = {"glass", "lamp_mirror", "wheel"}
ALWAYS_FUNCTIONAL   = {"severe_break"}   # severe_break on ANY part = functional

# ─── VLM (Vision-Language Model) ─────────────────────────────────────────────
VLM_ENABLED    = True
VLM_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

VLM_CONTEXT_PROMPT = """You are an expert vehicle damage assessor analyzing an image for an insurance claim system.
Respond EXACTLY in this format (one field per line, no extra text):

VEHICLES: <number of vehicles visible, e.g. 1, 2>
VEHICLE_TYPE: <primary vehicle type: 2W, 3W, or 4W>
VEHICLE_ANGLE: <viewing angle: front, rear, left_side, right_side, front_left, front_right, rear_left, rear_right, top, unknown>
DAMAGE_VISIBLE: <YES or NO>
DAMAGE_COUNT: <number of distinct damage areas visible, 0 if none>
DAMAGE_TYPES: <comma-separated from: scratch, dent, crack, broken_part, missing_part, paint_damage, deformation, shattered_glass, none>
DAMAGE_LOCATIONS: <comma-separated affected areas, e.g. front_bumper, fuel_tank, left_fender, rear_door, headlight, or none>
DAMAGE_SEVERITY: <overall: minor, moderate, severe, or none>
EXPOSED_MECHANICALS: <YES if engine/chain/suspension/frame are normally visible (e.g. motorcycle), NO if vehicle should be fully panelled>
PAINT_CONDITION: <good, faded, chipped, peeling, discolored, or unknown>
LIGHTING: <daylight, overcast, night, indoor, studio, mixed>
SURFACE: <comma-separated from: clean, muddy, wet, dusty, reflective, matte, rusty, none>

Rules:
- For 2W (motorcycles/scooters): exposed engine, chain, suspension, and frame are NORMAL, not damage.
- Only report damage you can clearly see — do not guess.
- DAMAGE_LOCATIONS should use specific part names, not generic terms."""

VLM_RATIONALE_PROMPT = """You are a senior vehicle damage assessor writing a professional report for an insurance claims adjuster.

IMAGE ANALYSIS (from AI system):
{damages_summary}

Based on BOTH the image and the AI analysis above, write a structured assessment:

1. DAMAGE SUMMARY: What damage is visible, where exactly, and how severe.
2. FUNCTIONAL IMPACT: Does this damage affect vehicle safety, drivability, or structural integrity?
3. CAVEATS: Any limitations in the assessment (image quality, obstructed views, uncertain areas).
4. RECOMMENDATION: Suggest next steps (approve claim, request more photos, in-person inspection).

Keep it professional, factual, and concise (4-6 sentences total). Do not fabricate damage that isn't visible."""

