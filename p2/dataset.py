import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from preprocessing import load_json_data, extract_chimneys, get_image_metadata


class ChimneyDataset(Dataset):
    def __init__(self, img_dir, label_dir, processor, transform=None):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.processor = processor
        self.transform = transform

        self.samples = []
        json_files = list(self.label_dir.glob("*.json"))[:100]  # limit for testing
        for i, json_path in enumerate(json_files):
            img_path = self.img_dir / json_path.with_suffix(".jpg").name
            if not img_path.exists():
                continue
            print(f"\rprocessing {img_path.name}, {i + 1}/{len(json_files)}", end="")
            # why does this take so long???
            json_data = load_json_data(json_path)
            chimneys = extract_chimneys(json_data)
            img_metadata = get_image_metadata(json_data)

            for chimney in chimneys:
                self.samples.append(
                    {
                        "img_path": img_path,
                        "chimney": chimney,
                        "img_metadata": img_metadata,
                    }
                )
        print(f"Loaded {len(self.samples)} chimney samples from {img_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample["img_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        metadata = torch.tensor(
            [
                sample["chimney"]["x_norm"],
                sample["chimney"]["y_norm"],
                sample["img_metadata"]["resolution"],
                sample["img_metadata"]["roll_tilt"],
                sample["img_metadata"]["pitch_tilt"],
            ],
            dtype=torch.float32,
        )

        target = torch.tensor(sample["chimney"]["height"], dtype=torch.float32)

        return pixel_values, metadata, target
