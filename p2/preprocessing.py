import json
from pathlib import Path


def load_json_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    key = list(data.keys())[0]
    return data[key]


def extract_chimneys(json_data):
    chimneys = []
    for region in json_data["regions"]:
        shape = region["shape_attributes"]
        attrs = region["region_attributes"]

        x_coords = shape["all_points_x"]
        y_coords = shape["all_points_y"]
        x_center = sum(x_coords) / len(x_coords)
        y_center = sum(y_coords) / len(y_coords)

        img_height = int(json_data["file_attributes"]["img_height"])
        img_width = int(json_data["file_attributes"]["img_width"])

        chimneys.append(
            {
                "chi_id": attrs["chi_id"],
                "height": float(attrs["chi_height_m"]),
                "x_norm": x_center / img_width,
                "y_norm": y_center / img_height,
            }
        )

    return chimneys


def get_image_metadata(json_data):
    attrs = json_data["file_attributes"]
    return {
        "resolution": float(attrs["img_resolution"]),
        "roll_tilt": float(attrs["img_roll_tilt"]),
        "pitch_tilt": float(attrs["img_pitch_tilt"]),
    }
