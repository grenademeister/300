import shutil
from pathlib import Path

def setup_data(base_data_dir="../../data", output_dir="data_yolo"):
    base_dir = Path(base_data_dir)
    out_dir = Path(output_dir)
    
    # Create directories
    dirs = {
        'train_labels': out_dir / "labels" / "train",
        'train_images': out_dir / "images" / "train", 
        'val_labels': out_dir / "labels" / "val",
        'val_images': out_dir / "images" / "val"
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Copy files
    sources = {
        'train_labels': (base_dir / "TL_KS_BBOX", "*.json"),
        'train_images': (base_dir / "TS_KS", "*.jpg"),
        'val_labels': (base_dir / "VL_KS_BBOX", "*.json"), 
        'val_images': (base_dir / "VS_KS", "*.jpg")
    }
    
    for key, (src_dir, pattern) in sources.items():
        if src_dir.exists():
            files = list(src_dir.glob(pattern))
            for file in files:
                shutil.copy2(file, dirs[key] / file.name)
            print(f"Copied {len(files)} {pattern} files to {dirs[key]}")
    
    return {**{k: str(v) for k, v in dirs.items()}, 'output_dir': str(out_dir)}

if __name__ == "__main__":
    setup_data()