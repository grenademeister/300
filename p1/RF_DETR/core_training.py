import os
from pathlib import Path


def train_rfdetr(
    dataset_path="/workspace/data_rf",
    output_path="/workspace/rfdetr_run",
    model_type="large",
    epochs=50,
    batch_size=6,
    grad_accum_steps=4,
    lr=1e-4,
):

    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    os.makedirs(output_path, exist_ok=True)

    # Import and initialize model
    if model_type.lower() == "large":
        from rfdetr.detr import RFDETRLarge

        model = RFDETRLarge()
    elif model_type.lower() == "base":
        from rfdetr.detr import RFDETRBase

        model = RFDETRBase()
    elif model_type.lower() == "small":
        from rfdetr.detr import RFDETRSmall

        model = RFDETRSmall()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Train model
    model.train(
        dataset_dir=dataset_path,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        lr=lr,
        output_dir=output_path,
    )

    return model
