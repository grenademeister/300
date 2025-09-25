import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
from utils import load_models

checkpoint = "PekingU/rtdetr_r50vd"
tmp = load_models(checkpoint)
image_processor = tmp["image_processor"]
train_transform = tmp["train_transformer"]
validation_transform = tmp["validation_transformer"]
model = tmp["model"]
evaluator = tmp["MAPEvaluator"]
from data_load import loading_data, RTDataset

paths = {
    "train_json": "/root/workspace/data/TL_KS_BBOX",
    "train_jpg": "/root/workspace/data/TS_KS",
    "validation_json": "/root/workspace/data/VL_KS_BBOX",
    "validation_jpg": "/root/workspace/data/VS_KS",
}
train_data = loading_data(paths["train_json"], paths["train_jpg"])

validation_data = loading_data(paths["validation_json"], paths["validation_jpg"])
train_RTdataset = RTDataset(train_data, image_processor, transform=train_transform)

validation_RTdataset = RTDataset(
    validation_data, image_processor, transform=validation_transform
)
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
import torch
from data_load import collate_fn


out_dir = "/workspace/RT_50"
training_args = TrainingArguments(
    output_dir=out_dir,
    num_train_epochs=60,
    max_grad_norm=0.1,
    learning_rate=1e-4,
    warmup_steps=300,
    per_device_train_batch_size=24,
    dataloader_num_workers=2,
    metric_for_best_model="eval_map",
    greater_is_better=True,
    load_best_model_at_end=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=False,
    eval_do_concat_batches=False,
    report_to="tensorboard",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_RTdataset,
    eval_dataset=validation_RTdataset,
    processing_class=image_processor,
    data_collator=collate_fn,
    compute_metrics=evaluator,
)

trainer.train()
