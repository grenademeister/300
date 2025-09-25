import torch
from torch import Tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def box_iou(box1, box2):
    """Compute IoU between box1 and box2"""
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    xx1 = torch.max(box1[:, 0].unsqueeze(1), box2[:, 0].unsqueeze(0))
    yy1 = torch.max(box1[:, 1].unsqueeze(1), box2[:, 1].unsqueeze(0))
    xx2 = torch.min(box1[:, 2].unsqueeze(1), box2[:, 2].unsqueeze(0))
    yy2 = torch.min(box1[:, 3].unsqueeze(1), box2[:, 3].unsqueeze(0))

    w = torch.clamp(xx2 - xx1, min=0)
    h = torch.clamp(yy2 - yy1, min=0)
    inter = w * h

    return inter / (area1.unsqueeze(1) + area2.unsqueeze(0) - inter)


class MetricEvaluator:
    def __init__(self):
        self.metric = MeanAveragePrecision(box_format="xywh", iou_thresholds=[0.5])

    def process_preds(self, pred: list[tuple[Tensor, Tensor]], type="xyxy"):
        """
        change box format
        Args:
            pred list(tuple): (Tensor(num_boxes, 4), Tensor(num_boxes,))
            type (str): box format, "xyxy" or "xywh"
        """
        out = []
        for boxes, scores in pred:
            if type == "xyxy" and self.metric.box_format == "xywh":
                x1, y1, x2, y2 = boxes.unbind(-1)
                boxes = torch.stack([x1, y1, x2 - x1, y2 - y1], dim=-1)
            elif type == "xywh" and self.metric.box_format == "xyxy":
                x, y, w, h = boxes.unbind(-1)
                boxes = torch.stack([x, y, x + w, y + h], dim=-1)
            out.append(
                {
                    "boxes": boxes,
                    "scores": scores,
                    "labels": torch.zeros(len(boxes), dtype=torch.int64),
                }
            )
        print(out)
        return out

    def update(self, preds, targets):
        self.metric.update(preds, targets)

    def compute(self):
        return self.metric.compute()


if __name__ == "__main__":
    # Test box_iou function
    box1 = torch.tensor([[0, 0, 2, 2], [1, 1, 3, 3]], dtype=torch.float32)
    box2 = torch.tensor([[1, 1, 2, 2], [0, 0, 1, 1]], dtype=torch.float32)
    iou = box_iou(box1, box2)
    print("IoU test:", iou)

    # Test MetricEvaluator
    evaluator = MetricEvaluator()

    # Sample predictions (boxes in xyxy format, scores)
    preds = [
        (torch.tensor([[10, 10, 20, 20]]), torch.tensor([0.9])),
        (torch.tensor([[30, 30, 40, 40]]), torch.tensor([0.2])),
    ]

    # Sample targets
    targets = [
        {"boxes": torch.tensor([[10, 10, 13, 13]]), "labels": torch.tensor([0])},
        {"boxes": torch.tensor([[30, 30, 10, 10]]), "labels": torch.tensor([0])},
    ]
    # Process predictions and update metric
    processed_preds = evaluator.process_preds(preds, type="xyxy")
    evaluator.update(processed_preds, targets)

    # Compute metrics
    result = evaluator.compute()
    print("mAP result:", result["map_50"])
