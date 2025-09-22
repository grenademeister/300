import torch


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


if __name__ == "__main__":
    box1 = torch.tensor([[0, 0, 2, 2], [1, 1, 3, 3]], dtype=torch.float32)
    box2 = torch.tensor([[1, 1, 2, 2], [0, 0, 1, 1]], dtype=torch.float32)
    iou = box_iou(box1, box2)
    print(iou)
    # Expected output:
    # tensor([[0.2500, 0.2500], [0.2500, 0.0000]])
