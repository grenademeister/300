from PIL import Image
import torch
from torch import Tensor
from torchvision import transforms


def tta(images: list[str]) -> list[Tensor]:
    """
    Apply test time augmentation
    Args:
        images (list[str]): List of image file paths
    Returns:
        list[torch.Tensor]: List of augmented image tensors
    """
    augmented_images = []
    for img in images:
        Image_img = Image.open(img).convert("RGB")
        Tensor_img = transforms.ToTensor()(Image_img)
        augmented_images.append(Tensor_img)
    batch_images = torch.stack(augmented_images, dim=0)

    return [
        batch_images,
        batch_images.flip(-1),  # H flip
        batch_images.flip(-2),  # V flip
        batch_images.flip(-1).flip(-2),  # HV flip
    ]


def reverse_tta(boxes: Tensor, scores: Tensor, tta_idx: int) -> tuple[Tensor, Tensor]:
    """
    Correct boxes position after TTA
    Args:
        boxes(torch.Tensor): Detected boxes, shape (batch, num_boxes, 4)
        scores(torch.Tensor): Confidence scores, shape (batch, num_boxes)
        tta_idx(int): Index of TTA applied
    """
    h, w = 512, 512

    boxes_corrected = boxes.clone()

    if tta_idx == 1:
        # H flip: x coordinates need correction
        boxes_corrected[:, :, [0, 2]] = w - boxes[:, :, [2, 0]]
    elif tta_idx == 2:
        # V flip: y coordinates need correction
        boxes_corrected[:, :, [1, 3]] = h - boxes[:, :, [3, 1]]
    elif tta_idx == 3:
        # HV flip: both x and y coordinates need correction
        boxes_corrected[:, :, [0, 2]] = w - boxes[:, :, [2, 0]]
        boxes_corrected[:, :, [1, 3]] = h - boxes[:, :, [3, 1]]

    return boxes_corrected, scores
