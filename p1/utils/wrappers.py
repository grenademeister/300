import torch
from torch import Tensor
import rfdetr
from supervision import Detections
from ultralytics import YOLO

from p1.utils.ensemble_boxes import nms, soft_nms, wbf
from p1.utils.augmentation import tta, reverse_tta, reverse_tta_new


class ModelWrapper:
    """
    Base class for model wrappers.
    Args:
        method (str): Ensemble method, one of ['nms', 'soft-nms', 'wbf']
        use_tta (bool): Whether to use test-time augmentation
    """

    def __init__(self, method: str = "nms", use_tta: bool = True):
        assert method in [
            "nms",
            "soft-nms",
            "wbf",
        ], "Invalid ensemble method."
        from typing import Callable

        self.ensemble_method: Callable = {"nms": nms, "soft-nms": soft_nms, "wbf": wbf}[
            method
        ]
        self.use_tta = use_tta

    def preprocess(
        self, images: list[str], labels: list[str]
    ) -> tuple[list[str], list[str]]:
        # Implement any preprocessing steps if necessary
        return images, labels

    def load_model(self):
        raise NotImplementedError

    def predict(self, images: list[str]) -> list[tuple[Tensor, Tensor]]:
        """Run full prediction pipeline: preprocess, inference, ensemble"""
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def inference(self, images: list[str]) -> tuple[Tensor, Tensor]:
        """Run only model inference"""
        raise NotImplementedError


class YOLOWrapper(ModelWrapper):
    def __init__(self, method="nms", use_tta=True):
        super().__init__(method, use_tta)
        self.model = self.load_model()

    def load_model(self, model_path="yolo_best.pt"):
        return YOLO(model_path)

    def inference(self, images: list[str]) -> tuple[list[Tensor], list[Tensor]]:
        """
        Run model inference.
        Args:
            images (list of str): List of image file paths, processed in batch
        Returns:
            results list[Tensor]: Detected boxes as list, tensor shape (total_num_boxes, 4)
            scores list[Tensor]: Respectively scores as list, tensor shape (total_num_boxes,)
        """
        if self.use_tta:
            images_input = tta(images)
        else:
            images_input = [images]
        # results is a list of input images in this point!
        results = []
        scores = []
        for i, image_batch in enumerate(images_input):  # loop over tta variation
            results_batch = self.model.predict(image_batch, verbose=False)
            temp_res, temp_scores = [], []
            for res in results_batch:
                temp_res.append(res.boxes.xyxy)
                temp_scores.append(res.boxes.conf)
            temp_res, temp_scores = reverse_tta_new(temp_res, temp_scores, i)
            results.append(temp_res)  # list[list[Tensor(num_boxes, 4)]]
            scores.append(temp_scores)  # list[list[Tensor(num_boxes)]]
        # concat results
        final_results = [torch.cat(r, 0) for r in zip(*results)]
        final_scores = [torch.cat(s, 0) for s in zip(*scores)]
        return final_results, final_scores

    def predict(self, images):
        results, scores = self.inference(images)
        final = self.ensemble_method(results, scores)
        return final

    def validate(self):
        # Implement validation logic if necessary
        pass


class RFWrapper(ModelWrapper):
    def __init__(self, method="nms", use_tta=True):
        super().__init__(method, use_tta)
        self.model = self.load_model()

    def load_model(self, model_path="checkpoint_best_ema.pth"):
        from rfdetr.detr import RFDETRLarge
        import torch.nn as nn

        model = RFDETRLarge(pretrain_weights=model_path)
        # model.optimize_for_inference()
        return model

    def preprocess(self, images_input: list[Tensor]) -> list[list[Tensor]]:
        images = []
        for image_batch in images_input:
            images.append(list(torch.unbind(image_batch, dim=0)))
        return images

    def inference(self, images: list[str]) -> tuple[list[Tensor], list[Tensor]]:
        """
        Run model inference.
        Args:
            images (list of str): List of image file paths, processed in batch
        Returns:
            results list[Tensor]: Detected boxes as list, tensor shape (total_num_boxes, 4)
            scores list[Tensor]: Respectively scores as list, tensor shape (total_num_boxes,)
        """
        if self.use_tta:
            images_input = tta(images)
        else:
            images_input = tta(images)[0:1]  # keep original only
        images_input = self.preprocess(images_input)
        # results is a list of input images in this point!
        results = []
        scores = []
        for i, image_batch in enumerate(images_input):
            results_batch = self.model.predict(image_batch)  # type: ignore
            temp_res, temp_scores = [], []
            if isinstance(results_batch, Detections):
                results_batch = [results_batch]  # << sick hack lol

            for detection in results_batch:
                temp_res.append(torch.from_numpy(detection.xyxy))  # (num_boxes, 4)
                temp_scores.append(torch.Tensor(detection.confidence))  # (num_boxes,)
            temp_res, temp_scores = reverse_tta_new(temp_res, temp_scores, i)
            results.append(temp_res)
            scores.append(temp_scores)
        # concat results
        final_results = [torch.cat(r, 0) for r in zip(*results)]
        final_scores = [torch.cat(s, 0) for s in zip(*scores)]
        return final_results, final_scores

    def predict(self, images):
        results, scores = self.inference(images)
        final = self.ensemble_method(results, scores)
        return final


class EnsembleWrapper(ModelWrapper):
    """
    Ensemble multiple model wrappers.
    """

    def __init__(
        self,
        model_wrappers: list[ModelWrapper],
        method: str = "nms",
        use_tta: bool = True,
    ):
        """
        Initialize the ensemble wrapper with multiple model wrappers.
        Args:
            model_wrappers (list of ModelWrapper): List of model wrapper instances to ensemble!
            method (str): Ensemble method, one of ['nms', 'soft-nms', 'wbf']
            use_tta (bool): Whether to use test-time augmentation
        """
        super().__init__(method, use_tta)
        self.load_model(model_wrappers)

    def load_model(self, model_wrappers):
        self.models: list[ModelWrapper] = model_wrappers

    def predict(self, images):
        """
        Stack predictions from multiple models and ensemble them.
        """
        results, scores = [], []
        for model in self.models:
            result, score = model.inference(images)
            results.append(result)
            scores.append(score)
            # result: list[Tensor(num_boxes, 4)]
            # score: list[Tensor(num_boxes,)]
        final_results = [torch.cat(r, 0) for r in zip(*results)]
        final_scores = [torch.cat(s, 0) for s in zip(*scores)]
        final = self.ensemble_method(final_results, final_scores)
        return final
