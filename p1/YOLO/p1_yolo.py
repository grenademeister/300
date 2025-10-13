from glob import glob
import json, os, yaml
from pathlib import Path
from ultralytics import YOLO
import shutil


def setup_data(base_data_dir="../../data", output_dir="data_yolo"):
    base_dir = Path(base_data_dir)
    out_dir = Path(output_dir)

    # Create directories
    dirs = {
        "train_labels": out_dir / "labels" / "train",
        "train_images": out_dir / "images" / "train",
        "val_labels": out_dir / "labels" / "val",
        "val_images": out_dir / "images" / "val",
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    # Copy files
    sources = {
        "train_labels": (base_dir / "TL_KS_BBOX", "*.json"),
        "train_images": (base_dir / "TS_KS", "*.jpg"),
        "val_labels": (base_dir / "VL_KS_BBOX", "*.json"),
        "val_images": (base_dir / "VS_KS", "*.jpg"),
    }

    for key, (src_dir, pattern) in sources.items():
        if src_dir.exists():
            files = list(src_dir.glob(pattern))
            for file in files:
                shutil.copy2(file, dirs[key] / file.name)
            print(f"Copied {len(files)} {pattern} files to {dirs[key]}")

    return {**{k: str(v) for k, v in dirs.items()}, "output_dir": str(out_dir)}


class YOLOWrapper:
    def __init__(self, data_dir="../../data", output_dir="data_yolo"):
        self.paths = setup_data(data_dir, output_dir)
        self.data_yaml = Path(self.paths["output_dir"]) / "dataset.yaml"

    def preprocess_json(self, path):
        files = glob("*.json", root_dir=path)
        print(f"Processing {len(files)} JSON files in {path}")

        for file in files:
            with open(os.path.join(path, file), "r", encoding="utf-8-sig") as f:
                data = json.load(f)
            item = list(data.values())[0]

            img_w = int(item["file_attributes"]["img_width"])
            img_h = int(item["file_attributes"]["img_height"])

            yolo_labels = []
            for obj in item["regions"]:
                x = obj["shape_attributes"]["x"]
                y = obj["shape_attributes"]["y"]
                w = obj["shape_attributes"]["width"]
                h = obj["shape_attributes"]["height"]
                cx = (x + w / 2) / img_w
                cy = (y + h / 2) / img_h
                nw = w / img_w
                nh = h / img_h
                yolo_labels.append(f"0 {cx} {cy} {nw} {nh}")

            out_file = file.replace(".json", ".txt")
            with open(os.path.join(path, out_file), "w") as f:
                f.write("\n".join(yolo_labels))

    def create_yaml(self):
        data_yaml = {
            "path": self.paths["output_dir"],
            "train": "images/train",
            "val": "images/val",
            "nc": 1,
            "names": ["chimney"],
        }

        with open(self.data_yaml, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

    def train(self, epochs=100, imgsz=512, batch=48, device=0):
        self.preprocess_json(self.paths["train_labels"])
        self.preprocess_json(self.paths["val_labels"])
        self.create_yaml()

        model = YOLO("yolo11x.pt")
        model.train(
            data=str(self.data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            optimizer="AdamW",
            project="YOLOv11-Experiments",
            exist_ok=True,
        )
        return model

    def validate(self, model):
        results = model.val(data=str(self.data_yaml))
        results_tta = model.val(data=str(self.data_yaml), augment=True)
        return results, results_tta


def main():
    wrapper = YOLOWrapper()
    model = wrapper.train()
    wrapper.validate(model)


if __name__ == "__main__":
    main()
