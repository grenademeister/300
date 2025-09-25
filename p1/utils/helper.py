# helper functions
from typing import Literal
import os
from glob import glob
from pathlib import Path
import json
import torch


def get_images_labels(purpose: Literal["train", "val"], data_dir="data", debug=False):
    """
    get image paths and label paths
    Args:
        purpose (str): "train" or "val"
        data_dir (str): root directory of data (I STRONGLY recommend to use your own data path.)
        debug (bool): if True, print some info
    Returns:
        image_paths (list of str): list of image file paths
        label_paths (list of str): list of label file paths
    """
    assert purpose in ["train", "val"], "Purpose must be 'train' or 'val'."
    data_subdir = {"train": "TS_KS", "val": "VS_KS"}[purpose]
    label_subdir = {"train": "TL_KS_BBOX", "val": "VL_KS_BBOX"}[purpose]

    image_paths = glob(os.path.join(data_dir, data_subdir, "*.jpg"))
    label_paths = glob(os.path.join(data_dir, label_subdir, "*.json"))
    # assert len(image_paths) == len(
    #     label_paths
    # ), f"Broken dataset: {len(image_paths)} images, {len(label_paths)} labels"
    # dataset is broken, fuck it.
    if len(image_paths) != len(label_paths):
        # Keep only overlapping files (by stem name)
        print(f"Broken dataset: {len(image_paths)} images, {len(label_paths)} labels")
        image_stems = {Path(p).stem for p in image_paths}
        label_stems = {Path(p).stem for p in label_paths}
        common_stems = image_stems & label_stems
        image_paths = [p for p in image_paths if Path(p).stem in common_stems]
        label_paths = [p for p in label_paths if Path(p).stem in common_stems]
        print(f"Only keeping {len(image_paths)} images and {len(label_paths)} labels")

    if debug:
        print(f"Directory {data_dir}, found {len(image_paths)} images")
    return image_paths, label_paths


def preprocess_json(label_paths):
    """
    preprocess json label files to get bounding boxes
    Args:
        label_paths (list of str): list of label file paths. use with get_images_labels()
    Returns:
        all_labels (list of dictionary): list of xywh boxes. processed for metricevaluator input format
    """
    all_labels = []
    for file in label_paths:
        with open(file, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        item = list(data.values())[0]

        img_w = int(item["file_attributes"]["img_width"])
        img_h = int(item["file_attributes"]["img_height"])

        labels = []
        for obj in item["regions"]:
            x = obj["shape_attributes"]["x"]
            y = obj["shape_attributes"]["y"]
            w = obj["shape_attributes"]["width"]
            h = obj["shape_attributes"]["height"]
            labels.append([x, y, w, h])
        labels = torch.tensor(labels, dtype=torch.float32)
        classes = torch.zeros(len(labels), dtype=torch.int64)  # dummy classes
        all_labels.append({"boxes": labels, "labels": classes})  # (num_boxes, 4)
    return all_labels  # list of Tensors
