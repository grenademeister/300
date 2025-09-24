from PIL import Image
import torch
from torch import Tensor
from torchvision import transforms


def get_image_size(image: str) -> tuple[int, int]:
    """
    Get image size
    Args:
        image (str): Image file path
    Returns:
        tuple[int, int]: Image size (height, width)
    """
    img = Image.open(image).convert("RGB")
    return img.size[1], img.size[0]  # (height, width)


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


def reverse_tta(
    boxes: Tensor, scores: Tensor, imgsz: tuple[int, int]
) -> tuple[Tensor, Tensor]:
    """
    Correct boxes position after TTA
    Args:
        boxes(torch.Tensor): Detected boxes, shape (batch, tta, num_boxes, 4) , required format is (xmin ymin xmax ymax)
        scores(torch.Tensor): Confidence scores, shape (batch, tta, num_boxes)
        imgsz (tuple[int, int]): Original image size (height, width)
    """
    h, w = imgsz

    # TTA order: [original, h_flip, v_flip, hv_flip]
    boxes_corrected = boxes.clone()

    # H flip: x coordinates need correction
    boxes_corrected[:, 1, :, [0, 2]] = w - boxes[:, 1, :, [2, 0]]

    # V flip: y coordinates need correction
    boxes_corrected[:, 2, :, [1, 3]] = h - boxes[:, 2, :, [3, 1]]

    # HV flip: both x and y coordinates need correction
    boxes_corrected[:, 3, :, [0, 2]] = w - boxes[:, 3, :, [2, 0]]
    boxes_corrected[:, 3, :, [1, 3]] = h - boxes[:, 3, :, [3, 1]]

    return boxes_corrected, scores
