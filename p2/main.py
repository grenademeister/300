from train import train_model
from model import ModelWrapper
from dataset import ChimneyDataset
import torch


def demo(
    fusion_type="baseline",
    img_dir="../data/TS_KS_new",
    label_dir="../data/TS_KS_LINE_new",
):
    print(f"=== Training {fusion_type} model ===")
    model_wrapper = train_model(
        img_dir=img_dir,
        label_dir=label_dir,
        epochs=10,
        batch_size=2,
        lr=1e-3,
        fusion_type=fusion_type,
    )

    print(f"\nInference with {fusion_type} model...")
    dataset = ChimneyDataset(img_dir, label_dir, model_wrapper.processor)
    pixel_values, metadata, target = dataset[0]

    prediction = model_wrapper.predict(pixel_values.unsqueeze(0), metadata.unsqueeze(0))

    print(f"True: {target.item():.2f}m, Predicted: {prediction.item():.2f}m\n")


if __name__ == "__main__":
    demo("baseline")
    # demo("cross_attn")
    # demo("film")
