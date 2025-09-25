from transformers.models.auto.image_processing_auto import AutoImageProcessor
from transformers.models.auto.modeling_auto import AutoModelForObjectDetection
import albumentations as A
import numpy as np
from dataclasses import dataclass
from transformers.image_transforms import center_to_corners_format
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch


def load_models(checkpoint) -> dict:
    """
    return: {
    image_processor: ~
    train_transformer:~
    validation_transformer:~
    model:~
    MAPEvaluator:~
    }

    """

    image_processor = AutoImageProcessor.from_pretrained(
        checkpoint,
        # do_resize=True,
        # size={"width": image_size, "height": image_size},
        use_fast=True,
    )

    train_augmentation_and_transform = A.Compose(
        [
            # A.Perspective(p=1),  # 원근왜곡
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),  # 이미지 밝기 조절
            A.GaussNoise(p=0.5),
            # A.HueSaturationValue(p=1), # 색상, 채도 명도 조절
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["category"],
            clip=True,
            min_area=25,
            min_width=1,
            min_height=1,
        ),
    )

    # to make sure boxes are clipped to image size and there is no boxes with area < 1 pixel
    validation_transform = A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["category"],
            clip=True,
            min_area=1,
            min_width=1,
            min_height=1,
        ),
    )
    id2label = {0: "chimeny"}
    label2id = {"chimeny": 0}

    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    eval_compute_metrics_fn = MAPEvaluator(
        image_processor=image_processor, threshold=0.01, id2label=id2label
    )
    return {
        "image_processor": image_processor,
        "train_transformer": train_augmentation_and_transform,
        "validation_transformer": validation_transform,
        "model": model,
        "MAPEvaluator": eval_compute_metrics_fn,
    }


@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


class MAPEvaluator:

    def __init__(self, image_processor, threshold=0.00, id2label=None):
        self.image_processor = image_processor
        self.threshold = threshold
        self.id2label = id2label

    def collect_image_sizes(self, targets):
        """Collect image sizes across the dataset as list of tensors with shape [batch_size, 2]."""
        image_sizes = []
        for batch in targets:
            batch_image_sizes = torch.tensor(np.array([x["size"] for x in batch]))
            image_sizes.append(batch_image_sizes)
        return image_sizes

    def collect_targets(self, targets, image_sizes):
        post_processed_targets = []
        for target_batch, image_size_batch in zip(targets, image_sizes):
            for target, size in zip(target_batch, image_size_batch):

                # here we have "yolo" format (x_center, y_center, width, height) in relative coordinates 0..1
                # and we need to convert it to "pascal" format (x_min, y_min, x_max, y_max) in absolute coordinates
                height, width = size
                boxes = torch.tensor(target["boxes"])
                boxes = center_to_corners_format(boxes)
                boxes = boxes * torch.tensor([[width, height, width, height]])

                labels = torch.tensor(target["class_labels"])
                post_processed_targets.append({"boxes": boxes, "labels": labels})
        return post_processed_targets

    def collect_predictions(self, predictions, image_sizes):
        post_processed_predictions = []
        for batch, target_sizes in zip(predictions, image_sizes):
            batch_logits, batch_boxes = batch[1], batch[2]
            output = ModelOutput(
                logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes)
            )
            post_processed_output = self.image_processor.post_process_object_detection(
                output, threshold=self.threshold, target_sizes=target_sizes
            )
            post_processed_predictions.extend(post_processed_output)
        return post_processed_predictions

    @torch.no_grad()
    def __call__(self, evaluation_results):

        predictions, targets = (
            evaluation_results.predictions,
            evaluation_results.label_ids,
        )

        image_sizes = self.collect_image_sizes(targets)
        post_processed_targets = self.collect_targets(targets, image_sizes)
        post_processed_predictions = self.collect_predictions(predictions, image_sizes)

        evaluator = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
        evaluator.warn_on_many_detections = False
        evaluator.update(post_processed_predictions, post_processed_targets)

        metrics = evaluator.compute()

        # Replace list of per class metrics with separate metric for each class
        classes = metrics.pop("classes")
        map_per_class = metrics.pop("map_per_class")
        mar_100_per_class = metrics.pop("mar_100_per_class")
        if classes.ndim == 0:
            classes = [classes]
        if map_per_class.ndim == 0:
            map_per_class = [map_per_class]
        if mar_100_per_class.ndim == 0:
            mar_100_per_class = [mar_100_per_class]
        for class_id, class_map, class_mar in zip(
            classes, map_per_class, mar_100_per_class
        ):
            class_name = (
                self.id2label[class_id.item()]
                if self.id2label is not None
                else class_id.item()
            )
            metrics[f"map_{class_name}"] = class_map
            metrics[f"mar_100_{class_name}"] = class_mar

        metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

        return metrics


class TTA:
    def __init__(self):
        """
        클래스 내부에서 적용할 증강(augmentation) 방식들을 직접 정의합니다.
        """
        # 적용할 증강의 종류를 딕셔너리로 정의
        self.augmentations = {
            "flip": A.HorizontalFlip(p=1),
            "intensity": A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=1
            ),
            "gaussian_noise": A.GaussNoise(p=1),
        }

        # 각 증강을 A.Compose로 감싸서 개별 파이프라인으로 만듭니다.
        self.transforms = {
            name: A.Compose([aug]) for name, aug in self.augmentations.items()
        }

    def __call__(self, batch_tensor: torch.Tensor) -> dict:
        """
        데이터로더에서 나온 배치 텐서에 TTA를 적용하고 딕셔너리를 반환합니다.

        Args:
            batch_tensor (torch.Tensor): (B, C, H, W) 형태의 이미지 텐서.

        Returns:
            dict: {'original': 원본 텐서, 'aug_name1': 증강 텐서 1, ...} 형태의 딕셔너리.
        """
        # 원본 텐서를 'original' 키와 함께 결과 딕셔너리에 추가합니다.
        results = {"original": batch_tensor}
        device = batch_tensor.device

        # PyTorch 텐서(B, C, H, W)를 NumPy 배열(B, H, W, C)로 변환합니다.
        batch_numpy = batch_tensor.permute(0, 2, 3, 1).cpu().numpy()

        # __init__에서 정의된 각 증강을 이름(aug_name)과 함께 순회합니다.
        for aug_name, transform in self.transforms.items():

            # 배치 내의 각 이미지에 개별적으로 증강을 적용합니다.
            augmented_batch_list = [
                transform(image=img)["image"] for img in batch_numpy
            ]

            # 증강된 이미지들을 다시 하나의 NumPy 배열로 합칩니다.
            augmented_batch_numpy = np.array(augmented_batch_list)

            # NumPy 배열을 다시 PyTorch 텐서로 변환하고 원본 장치로 보냅니다.
            augmented_tensor = (
                torch.from_numpy(augmented_batch_numpy).permute(0, 3, 1, 2).to(device)
            )

            # 증강 이름(key)과 증강된 텐서(value)를 딕셔ner셔너리에 추가합니다.
            results[aug_name] = augmented_tensor

        return results
