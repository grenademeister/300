import os
import shutil
import glob


def setup_directories(base_path="/workspace/data_rf"):
    os.makedirs(f"{base_path}/train", exist_ok=True)
    os.makedirs(f"{base_path}/valid", exist_ok=True)
    os.makedirs(f"{base_path}/test", exist_ok=True)


def copy_data(source_path="/workspace/data", target_path="/workspace/data_rf"):
    # Training data
    shutil.copytree(f"{source_path}/TS_KS", f"{target_path}/train", dirs_exist_ok=True)
    shutil.copytree(f"{source_path}/TL_KS_BBOX", f"{target_path}/train", dirs_exist_ok=True)
    
    # Validation data
    shutil.copytree(f"{source_path}/VS_KS", f"{target_path}/valid", dirs_exist_ok=True)
    shutil.copytree(f"{source_path}/VL_KS_BBOX", f"{target_path}/valid", dirs_exist_ok=True)
    
    # Test data (copy from valid)
    shutil.copytree(f"{target_path}/valid", f"{target_path}/test", dirs_exist_ok=True)


def clean_annotations(target_path="/workspace/data_rf"):
    for f in glob.glob(f"{target_path}/*/_annotations.coco.json"):
        os.remove(f)


def prepare_dataset(source_path="/workspace/data", target_path="/workspace/data_rf", output_path="/workspace/rfdetr_run"):
    setup_directories(target_path)
    os.makedirs(output_path, exist_ok=True)
    copy_data(source_path, target_path)
    clean_annotations(target_path)