import torch

from p1.utils.ensemble_boxes import nms, soft_nms, wbf
from p1.utils.augmentation import tta


class ModelWrapper:
    def __init__(self, method="nms", use_tta=True):
        self.model = self.load_model()
        assert method in ["nms", "soft-nms", "wbf"], "Invalid method"
        if method == "nms":
            self.ensemble_method = nms
        elif method == "soft-nms":
            self.ensemble_method = soft_nms
        else:
            self.ensemble_method = wbf
        self.use_tta = use_tta

    def preprocess(self, images, labels):
        # Implement any preprocessing steps if necessary
        return images, labels

    def load_model(self):
        raise NotImplementedError

    def predict(self, images):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError


class YOLOWrapper(ModelWrapper):
    def __init__(self, method="nms", use_tta=True):
        super().__init__(method, use_tta)

    def load_model(self, model_path="yolo_best.pt"):
        from ultralytics import YOLO

        return YOLO(model_path)

    def predict(self, images):
        """
        This function is IMPORTANT! read before adding new Wrappers!
        """
        if self.use_tta:
            images_input = tta(images)
        else:
            images_input = [images]
        # results is a list of input images in this point!
        results = []
        scores = []
        for image_batch in images_input:
            results_batch = self.model.predict(image_batch)
            temp_res, temp_scores = [], []
            for res in results_batch:
                temp_res.append(res.boxes.xyxy)
                temp_scores.append(res.boxes.conf)
            results.append(torch.stack(temp_res, dim=0))
            scores.append(torch.stack(temp_scores, dim=0))
        # make results a tensor
        results = torch.stack(results, dim=0).permute(1, 0, 2, 3)
        scores = torch.stack(scores, dim=0).permute(1, 0, 2)
        # results: (batch, tta, num_boxes, 4)
        # scores: (batch, tta, num_boxes)

        final = self.ensemble_method(results, scores)
        return final

    def validate(self):
        # Implement validation logic if necessary
        pass
