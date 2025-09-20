import json
import math
from pathlib import Path


def preprocess_to_coco(pathname):
    DATA_DIR = Path(pathname)
    OUT = DATA_DIR / "_annotations.coco.json"
    
    images, annotations = [], []
    img_id, ann_id = 1, 1
    
    for jf in DATA_DIR.glob("*.json"):
        raw = json.load(open(jf, "r", encoding="utf-8"))
        entry = list(raw.values())[0]
        filename = entry["filename"]
        fa = entry["file_attributes"]
        width, height = int(fa["img_width"]), int(fa["img_height"])
        
        images.append({"id": img_id, "file_name": filename, "width": width, "height": height})
        
        for r in entry["regions"]:
            sa = r["shape_attributes"]
            if sa["name"] == "rect":
                x, y, w, h = map(int, (sa["x"], sa["y"], sa["width"], sa["height"]))
                
                # Clamp to image bounds
                x, y = max(0, x), max(0, y)
                w = min(w, width - x)
                h = min(h, height - y)
                
                if w > 0 and h > 0:
                    annotations.append({
                        "id": ann_id, "image_id": img_id, "category_id": 0,
                        "bbox": [x, y, w, h], "area": w * h,
                        "segmentation": [[x, y, x+w, y, x+w, y+h, x, y+h]], "iscrowd": 0
                    })
                    ann_id += 1
        img_id += 1
    
    coco = {
        "info": {}, "licenses": [], "images": images, "annotations": annotations,
        "categories": [{"id": 0, "name": "chimney", "supercategory": "none"}]
    }
    
    OUT.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(images), len(annotations)


def validate_coco(annotation_path):
    c = json.load(open(annotation_path, "r", encoding="utf-8"))
    imgs = {im["id"]: im for im in c["images"]}
    
    for ann in c["annotations"]:
        im = imgs[ann["image_id"]]
        w_img, h_img = im["width"], im["height"]
        x, y, w, h = ann["bbox"]
        
        if len(ann["bbox"]) != 4 or w <= 0 or h <= 0:
            return False
        if x < 0 or y < 0 or (x + w) > w_img or (y + h) > h_img:
            return False
        if any(math.isnan(v) or math.isinf(v) for v in (x, y, w, h)):
            return False
    
    return True


def process_annotations(base_path="/workspace/data_rf"):
    splits = ["train", "valid", "test"]
    stats = {}
    
    for split in splits:
        split_path = f"{base_path}/{split}"
        if Path(split_path).exists():
            img_count, ann_count = preprocess_to_coco(split_path)
            is_valid = validate_coco(f"{split_path}/_annotations.coco.json")
            stats[split] = {"images": img_count, "annotations": ann_count, "valid": is_valid}
        else:
            stats[split] = {"images": 0, "annotations": 0, "valid": False}
    
    return stats