"""
GlobalAnnotationDB — Extracted from the notebook for standalone pipeline usage.

Builds a unified annotation database from all 7 datasets:
  - COCO format (damage)
  - VIA format (damage)
  - Supervisely format (anatomy)

Provides O(1) lookup by image hash for polygon annotations.
"""
import os
import json
import numpy as np
from collections import defaultdict

from pipeline.config import (
    BASE, DAMAGE_CLASSES, DAMAGE_REMAP,
    ANATOMY_CLASSES, ANATOMY_REMAP,
)


class GlobalAnnotationDB:
    """
    Unified annotation registry.

    On construction, parses all JSON annotation files across all datasets
    and indexes them by image hash for O(1) lookup.

    Attributes:
        damage_db:  dict[hash → list of {class_id, points}]
        anatomy_db: dict[hash → list of {class_id, points}]
    """

    def __init__(self, manifests: list):
        self.damage_db  = defaultdict(list)
        self.anatomy_db = defaultdict(list)
        self.lookup     = {}       # (dataset, filename) → hash
        self.path_lookup = {}      #  image_path → hash

        for manifest in manifests:
            with open(manifest, "r") as f:
                data = json.load(f)
                for item in data:
                    key = (item["dataset"],
                           os.path.basename(item["image_path"]))
                    self.lookup[key] = item["hash"]
                    self.path_lookup[item["image_path"]] = item["hash"]

        self._parse_all_jsons()

    def _parse_all_jsons(self):
        print("Loading global annotation polygons into memory...")

        # ── Parse COCO & VIA (Damage) ─────────────────────────────────────
        for dataset in ["coco_annotated", "vehicle_damage",
                        "gdrive_dataset", "coco_car_damage"]:
            dataset_path = os.path.join(BASE, dataset)
            if not os.path.exists(dataset_path):
                continue

            for root, _, files in os.walk(dataset_path):
                for f in files:
                    if not f.endswith(".json"):
                        continue
                    full_path = os.path.join(root, f)
                    try:
                        with open(full_path) as jf:
                            data = json.load(jf)
                    except Exception:
                        continue

                    # COCO format
                    if isinstance(data, dict) and "annotations" in data:
                        self._parse_coco(data, dataset)

                    # VIA format
                    elif (isinstance(data, dict) and
                          all(isinstance(v, dict) for v in data.values())):
                        self._parse_via(data, dataset)

        # ── Parse Supervisely (Anatomy) ───────────────────────────────────
        anatomy_path = os.path.join(BASE, "car_parts_and_damages")
        anatomy_matched = 0
        anatomy_orphaned = 0

        if os.path.exists(anatomy_path):
            for root, _, files in os.walk(anatomy_path):
                for f in files:
                    if not f.endswith(".json") or f == "meta.json":
                        continue

                    filename_no_ext = f.replace(".json", "")
                    img_hash = self._find_hash("car_parts_and_damages",
                                                filename_no_ext)
                    if img_hash:
                        anatomy_matched += 1
                        try:
                            with open(os.path.join(root, f)) as jf:
                                data = json.load(jf)
                        except Exception:
                            continue

                        objects = data.get("objects", []) or data.get("tags", [])
                        for obj in objects:
                            raw_cls = (obj.get("classTitle") or
                                       obj.get("class") or
                                       obj.get("title"))
                            canonical = ANATOMY_REMAP.get(raw_cls)

                            if (canonical and
                                isinstance(obj.get("points"), dict) and
                                "exterior" in obj["points"]):
                                pts = np.array(obj["points"]["exterior"],
                                               dtype=np.int32)
                                if len(pts) > 2:
                                    self.anatomy_db[img_hash].append({
                                        "class_id": ANATOMY_CLASSES[canonical],
                                        "points": pts,
                                    })
                    else:
                        anatomy_orphaned += 1

        print(f"DB Built! Damage images: {len(self.damage_db)}")
        print(f"Anatomy images: {len(self.anatomy_db)}")
        print(f"Anatomy matched: {anatomy_matched}, "
              f"orphaned: {anatomy_orphaned}")

    def _find_hash(self, dataset: str, filename_no_ext: str):
        """Try multiple extensions to find the hash."""
        target = (dataset, filename_no_ext)
        if target in self.lookup:
            return self.lookup[target]
        for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG"]:
            key = (dataset, filename_no_ext + ext)
            if key in self.lookup:
                return self.lookup[key]
        return None

    def _parse_coco(self, data: dict, dataset: str):
        """Parse COCO-format annotations."""
        cat_map = {c["id"]: c.get("name", "")
                   for c in data.get("categories", [])}
        img_map = {img["id"]: os.path.basename(img["file_name"])
                   for img in data.get("images", [])}

        for ann in data["annotations"]:
            raw_cls = cat_map.get(ann["category_id"])
            canonical = DAMAGE_REMAP.get(raw_cls)
            filename = img_map.get(ann["image_id"])
            target = (dataset, filename)

            if (canonical and target in self.lookup and
                "segmentation" in ann):
                img_hash = self.lookup[target]
                for seg in ann["segmentation"]:
                    if isinstance(seg, list) and len(seg) > 4:
                        pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
                        self.damage_db[img_hash].append({
                            "class_id": DAMAGE_CLASSES[canonical],
                            "points": pts,
                        })

    def _parse_via(self, data: dict, dataset: str):
        """Parse VIA-format annotations."""
        for img_name, img_data in data.items():
            filename = os.path.basename(img_name)
            target = (dataset, filename)

            if "regions" not in img_data or target not in self.lookup:
                continue

            img_hash = self.lookup[target]
            for region in img_data["regions"]:
                raw_cls = region.get("class")
                canonical = DAMAGE_REMAP.get(raw_cls)

                if not canonical:
                    continue

                if ("shape_attributes" in region and
                    region["shape_attributes"].get("name") == "polygon"):
                    shape = region["shape_attributes"]
                    pts = np.column_stack(
                        (shape["all_x"], shape["all_y"])
                    ).astype(np.int32)
                    self.damage_db[img_hash].append({
                        "class_id": DAMAGE_CLASSES[canonical],
                        "points": pts,
                    })
                elif "all_x" in region and "all_y" in region:
                    pts = np.column_stack(
                        (region["all_x"], region["all_y"])
                    ).astype(np.int32)
                    self.damage_db[img_hash].append({
                        "class_id": DAMAGE_CLASSES[canonical],
                        "points": pts,
                    })
