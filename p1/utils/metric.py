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


def calculate_iou_matrix(self, bboxes1: torch.Tensor, bboxes2: torch.Tensor):
    """
    두 개의 바운딩 박스 배치 텐서 간의 IoU 행렬을 계산하고,
    각 정답(bboxes2) 박스에 대한 최대 IoU 값을 찾아 리스트로 반환합니다.

    Args:
        bboxes1 (torch.Tensor): 예측 바운딩 박스 텐서.
                                형태: (B, N, 4), [x, y, w, h]
        bboxes2 (torch.Tensor): 정답 바운딩 박스 텐서.
                                형태: (B, M, 4), [x, y, w, h]

    Returns:
        list: 길이가 B인 리스트.
            각 원소는 (M,) 형태의 텐서이며, 해당 배치의 각 정답 박스에 대한
            최대 IoU 값을 담고 있습니다.
    """
    # 입력 텐서의 데이터 타입을 float으로 통일
    bboxes1 = bboxes1.float()
    bboxes2 = bboxes2.float()

    # [x, y, w, h] -> [x1, y1, x2, y2] 형태로 변환
    boxes1 = torch.cat([bboxes1[..., :2], bboxes1[..., :2] + bboxes1[..., 2:]], dim=-1)
    boxes2 = torch.cat([bboxes2[..., :2], bboxes2[..., :2] + bboxes2[..., 2:]], dim=-1)

    # 각 박스의 면적 계산
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # 교차 영역(intersection) 계산
    inter_x1 = torch.maximum(boxes1[..., 0].unsqueeze(2), boxes2[..., 0].unsqueeze(1))
    inter_y1 = torch.maximum(boxes1[..., 1].unsqueeze(2), boxes2[..., 1].unsqueeze(1))
    inter_x2 = torch.minimum(boxes1[..., 2].unsqueeze(2), boxes2[..., 2].unsqueeze(1))
    inter_y2 = torch.minimum(boxes1[..., 3].unsqueeze(2), boxes2[..., 3].unsqueeze(1))

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)

    intersection_area = inter_w * inter_h

    # 합집합(union) 영역 계산
    union_area = area1.unsqueeze(2) + area2.unsqueeze(1) - intersection_area

    # IoU 행렬 계산. shape: (B, N, M)
    iou = intersection_area / (union_area + 1e-8)

    # --- 아래 부분이 추가/수정된 핵심 로직 ---

    # 1. 각 정답(label) 박스에 대해 가장 높은 IoU 값을 찾습니다.
    # dim=1은 예측(N) 차원을 의미합니다.
    # best_ious의 shape은 (B, M)이 됩니다.
    best_ious, _ = torch.max(iou, dim=1)

    # 2. 결과를 배치별로 분리하여 리스트로 만듭니다.
    result_list = [v for v in best_ious]
    result_list = [1 if v > 0.5 else 0 for v in result_list]
    return result_list


class MetricEvaluator:
    """
    Wrapper class for MAP from torchmetrics.\n
    Use process_preds to convert predictions to required format, then use update() and compute().\n
    Use process_json from helper if needed.
    """

    def __init__(self):
        self.metric = MeanAveragePrecision(box_format="xywh")

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
        # print(out)
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
        (torch.tensor([[30, 30, 40, 40]]), torch.tensor([0.8])),
    ]

    # Sample targets
    targets = [
        {"boxes": torch.tensor([[10, 10, 13, 13]]), "labels": torch.tensor([0])},
        {"boxes": torch.tensor([[30, 30, 40, 40]]), "labels": torch.tensor([0])},
    ]
    # Process predictions and update metric
    processed_preds = evaluator.process_preds(preds, type="xyxy")
    evaluator.update(processed_preds, targets)

    # Compute metrics
    result = evaluator.compute()
    print("mAP result:", result["map_50"])
