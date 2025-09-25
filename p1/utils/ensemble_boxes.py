import torch
from torchvision.ops import nms as torch_nms
from p1.utils.metric import box_iou


def nms(boxes, scores, iou_threshold=0.5):
    batch_size, num_boxes_total, _ = boxes.shape

    results = []
    for b in range(batch_size):
        valid_mask = scores[b] > 0
        if not valid_mask.any():
            results.append((torch.empty(0, 4), torch.empty(0)))
            continue

        valid_boxes = boxes[b][valid_mask]
        valid_scores = scores[b][valid_mask]

        keep = torch_nms(valid_boxes, valid_scores, iou_threshold)
        results.append((valid_boxes[keep], valid_scores[keep]))

    return results


def soft_nms(boxes, scores, iou_threshold=0.5, sigma=0.5, score_threshold=0.001):
    batch_size, num_boxes_total, _ = boxes.shape

    results = []
    for b in range(batch_size):
        valid_mask = scores[b] > score_threshold
        if not valid_mask.any():
            results.append((torch.empty(0, 4), torch.empty(0)))
            continue

        valid_boxes = boxes[b][valid_mask].clone()
        valid_scores = scores[b][valid_mask].clone()

        # IoU matrix computation
        x1 = valid_boxes[:, 0].unsqueeze(1)
        y1 = valid_boxes[:, 1].unsqueeze(1)
        x2 = valid_boxes[:, 2].unsqueeze(1)
        y2 = valid_boxes[:, 3].unsqueeze(1)

        areas = (x2 - x1) * (y2 - y1)

        xx1 = torch.max(x1, x1.t())
        yy1 = torch.max(y1, y1.t())
        xx2 = torch.min(x2, x2.t())
        yy2 = torch.min(y2, y2.t())

        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        inter = w * h

        iou = inter / (areas + areas.t() - inter)

        # Gaussian decay
        decay = torch.exp(-(iou**2) / sigma)
        decay[torch.eye(len(valid_scores)).bool()] = 1.0

        # Apply decay to scores
        for i in range(len(valid_scores)):
            valid_scores *= decay[valid_scores.argmax()]

        keep_mask = valid_scores > score_threshold
        results.append((valid_boxes[keep_mask], valid_scores[keep_mask]))

    return results


def wbf(boxes, scores, iou_threshold=0.5):
    batch_size, num_boxes_total, _ = boxes.shape

    results = []
    for b in range(batch_size):
        valid_mask = scores[b] > 0
        if not valid_mask.any():
            results.append((torch.empty(0, 4), torch.empty(0)))
            continue

        all_boxes = boxes[b][valid_mask]
        all_scores = scores[b][valid_mask]

        # Group overlapping boxes
        fused_boxes = []
        fused_scores = []
        used = torch.zeros(len(all_boxes), dtype=torch.bool)

        for i in range(len(all_boxes)):
            if used[i]:
                continue

            # Find overlapping boxes
            iou = box_iou(all_boxes[i : i + 1], all_boxes)
            overlap_mask = (iou > iou_threshold).squeeze(0)
            overlap_mask[i] = True

            if overlap_mask.sum() == 1:
                fused_boxes.append(all_boxes[i])
                fused_scores.append(all_scores[i])
            else:
                # Weighted fusion
                overlap_boxes = all_boxes[overlap_mask]
                overlap_scores = all_scores[overlap_mask]

                # Weighted average of coordinates
                weights_norm = overlap_scores / overlap_scores.sum()
                fused_box = (overlap_boxes * weights_norm.unsqueeze(1)).sum(0)
                fused_score = overlap_scores.mean()  ## mean()아니노? 맞음 ㅎ

                fused_boxes.append(fused_box)
                fused_scores.append(fused_score)

            used[overlap_mask] = True

        if fused_boxes:
            final_boxes = torch.stack(fused_boxes)
            final_scores = torch.stack(fused_scores)
            results.append((final_boxes, final_scores))
        else:
            results.append((torch.empty(0, 4), torch.empty(0)))

    return results
