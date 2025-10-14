import json
from pathlib import Path


def load_json_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def extract_chimneys(json_data):
    chimneys = []

    for file_id, item in json_data.items():
        fa = item.get("file_attributes", {}) or {}
        regions = item.get("regions", []) or []

        try:
            img_height = fa.get("img_height")
            img_width = fa.get("img_width")
            if img_height is None or img_width is None:
                continue
            img_h = int(float(img_height))
            img_w = int(float(img_width))
        except (TypeError, ValueError):
            continue

        for r in regions:
            shape = r.get("shape_attributes", {}) or {}
            attrs = r.get("region_attributes", {}) or {}

            chi_height = attrs.get("chi_height_m")
            if chi_height is None:
                continue
            try:
                height = float(chi_height)
            except (TypeError, ValueError):
                continue

            if "all_points_x" in shape and "all_points_y" in shape:
                xs = shape["all_points_x"]
                ys = shape["all_points_y"]
            else:
                continue

            if not (xs and ys):
                continue
            x_center = sum(xs) / len(xs)
            y_center = sum(ys) / len(ys)
            x_norm = x_center / img_w
            y_norm = y_center / img_h

            chimneys.append(
                {
                    "chi_id": attrs.get("chi_id"),
                    "height": height,
                    "x1": xs[0],
                    "x2": xs[1],
                    "y1": ys[0],
                    "y2": ys[1],
                    "file_id": file_id,
                }
            )

    return chimneys


def get_image_metadata(json_data, file_id=None):
    """
    json_data: (a) {file_id: item} 또는 (b) 단일 item(dict)
    file_id: extract_chimneys가 돌려준 chimney['file_id']를 넣으면 정확히 매칭
    반환: {'resolution': float, 'roll_tilt': float, 'pitch_tilt': float}
    """
    if isinstance(json_data, dict) and "regions" in json_data:
        item = json_data
    else:
        if not isinstance(json_data, dict):
            raise TypeError("get_image_metadata: dict 형태의 json_data가 필요합니다.")
        if file_id is not None:
            item = json_data.get(file_id, {})
        else:
            item = next(iter(json_data.values()), {})

    attrs = item.get("file_attributes", {}) or {}

    def _f(val, default=0.0):
        try:
            return float(val)
        except (TypeError, ValueError):
            return float(default)

    return {
        "resolution": _f(attrs.get("img_resolution"), 0.0),
        "roll_tilt": _f(attrs.get("img_roll_tilt"), 0.0),
        "pitch_tilt": _f(attrs.get("img_pitch_tilt"), 0.0),
    }
