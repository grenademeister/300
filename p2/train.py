import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from model import ModelWrapper
from dataset import ChimneyDataset


def train_epoch(model_wrapper, dataloader, optimizer, criterion):
    model_wrapper.train_mode()
    total_loss = 0

    for pixel_values, metadata, targets in dataloader:
        pixel_values = pixel_values.to(model_wrapper.device)
        metadata = metadata.to(model_wrapper.device)
        targets = targets.to(model_wrapper.device)

        optimizer.zero_grad()
        predictions = model_wrapper.model(pixel_values, metadata)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def eval_epoch(model_wrapper, dataloader, criterion):
    model_wrapper.eval_mode()
    total_loss = 0

    with torch.no_grad():
        for pixel_values, metadata, targets in dataloader:
            pixel_values = pixel_values.to(model_wrapper.device)
            metadata = metadata.to(model_wrapper.device)
            targets = targets.to(model_wrapper.device)

            predictions = model_wrapper.model(pixel_values, metadata)
            loss = criterion(predictions, targets)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def train_model(
    img_dir,
    label_dir,
    epochs=50,
    batch_size=8,
    lr=1e-3,
    val_split=0.2,
    fusion_type="baseline",
):
    model_wrapper = ModelWrapper(fusion_type=fusion_type)

    full_dataset = ChimneyDataset(img_dir, label_dir, model_wrapper.processor)

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    trainable_params = [p for p in model_wrapper.model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")

    for epoch in range(epochs):
        train_loss = train_epoch(model_wrapper, train_loader, optimizer, criterion)
        val_loss = eval_epoch(model_wrapper, val_loader, criterion)

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_wrapper.save("best_model.pth")

    return model_wrapper
