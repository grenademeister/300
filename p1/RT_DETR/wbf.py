from .data_load import RTDataset, collate_fn
from pathlib import Path
import numpy as np
import torch
from torch import Tensor
from transformers.models.auto.image_processing_auto import AutoImageProcessor
from transformers.models.auto.modeling_auto import AutoModelForObjectDetection
import albumentations as A
import os
import json
from datasets import Dataset, Image
from PIL import Image
from p1.utils.metric import calculate_iou_matrix
from p1.utils.wrappers import ModelWrapper
from p1.utils.augmentation import tta, reverse_tta, reverse_tta_new


class RTWrapper(ModelWrapper):
    def __init__(
        self,
        method="wbf",
        use_tta=True,
        checkpoint="/home/parkjunsu/workspace/300/checkpoint-9744",
        device="cuda",
    ):
        self.checkpoint = checkpoint
        self.device = device
        self.image_processor = AutoImageProcessor.from_pretrained(self.checkpoint)
        self.coco = A.Compose(
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
        super().__init__(method, use_tta)
        self.model = self.load_model()

    def preprocess(self):
        val_data = self.loading_data()
        val_data = RTDataset(val_data, self.image_processor, transform=self.coco)
        return val_data

    def load_model(self):
        return AutoModelForObjectDetection.from_pretrained(self.checkpoint)

    def inference(self, images) -> tuple[list[Tensor], list[Tensor]]:
        """
        arg: list(path)
        return: [ (bboxes, score) ,() ~]
        """
        inputs = []
        self.use_tta = True
        if self.use_tta:
            inputs = tta(images=images)
            #  print('after tta shpae' , len(inputs) , inputs[0].shape) ## 4 size [batch,3,512,512]
            for i in range(len(inputs)):
                inputs[i] = self.image_processor(
                    inputs[i], return_tensors="pt", do_rescale=False
                )["pixel_values"]
        else:
            for image in images:
                inputs.append(Image.open(image))

            r_inputs = []
            r_inputs.append(
                self.image_processor(inputs, return_tensors="pt")["pixel_values"]
            )
            inputs = r_inputs
        target_size = torch.tensor([512, 512]).unsqueeze(0).expand(len(images), -1)
        outputs = []
        for input in inputs:
            with torch.no_grad():
                outputs.append(
                    self.model(input)
                )  # model output: xmin ymin xmax ymax / tta-list ( batch-list ( ~))

        results = []
        for output in outputs:
            results.append(
                self.image_processor.post_process_object_detection(
                    output, threshold=0.25, target_sizes=target_size
                )
            )
        bboxes = [[d["boxes"] for d in batch] for batch in results]
        scores = [[d["scores"] for d in batch] for batch in results]

        for i, (bbox, score) in enumerate(zip(bboxes, scores)):
            bboxes[i], scores[i] = reverse_tta_new(bbox, score, i)

        final_results, final_scores = [], []
        final_results = [torch.cat(r, 0) for r in zip(*bboxes)]
        final_scores = [torch.cat(s, 0) for s in zip(*scores)]
        return final_results, final_scores

    def predict(self, images):
        bboxes, scores = self.inference(images)
        # print("before ensemble:")
        # print(bboxes)
        final = self.ensemble_method(bboxes, scores)
        # print("after ensemble:")
        # print(final)
        return final

    def result_process(self, r: list[list[dict[str, torch.Tensor]]]):
        """
        DEPRECATED
        Args:
            r(list[list[dict[str, torch.Tensor]]]): tta-list
        Returns:
            list-tta ( tensor ( batch ,num , 4)) , list-tta(tensor(batch,num,1))
        """
        return
        tta_num = len(r)
        batch_num = len(r[0])
        max_bbox_num = 0

        ## padding
        padded_b = [
            (
                torch.cat([t, torch.full((max_bbox_num - t.shape[0], 4), -1.0)])
                if t.shape[0] < max_bbox_num
                else t
            )
            for t in flat_b
        ]
        padded_s = [
            (
                torch.cat([t, torch.full((max_bbox_num - t.shape[0], 4), -1.0)])
                if t.shape[0] < max_bbox_num
                else t
            )
            for t in flat_s
        ]

        # reshape
        results_b = torch.stack(padded_b).view(tta_num, batch_num, max_bbox_num, 4)
        results_b = [b for b in results_b]
        results_s = torch.stack(padded_s).view(tta_num, batch_num, max_bbox_num, 1)
        results_s = [s for s in results_s]
        return results_b, results_s

    def validate(self):
        n = 2
        val_data = self.preprocess()
        val_data_loader = torch.utils.data.DataLoader(
            val_data, batch_size=n, collate_fn=collate_fn  # type: ignore
        )
        miou = []

        with torch.no_grad():
            for batch in val_data_loader:

                label = batch["labels"]
                output = self.model(batch["pixel_values"])
                target_size = torch.tensor([512, 512]).unsqueeze(0).expand(n, -1)
                output = self.image_processor.post_process_object_detection(
                    output, threshold=0.25
                )  ##output format : (xmin,ymin,xmax,ymax) , label format :(xcenter ycenter w h)

                output, label = self.post_process(output, label)
                i = calculate_iou_matrix(output, label)

                # if i.ndim == 0:
                #     miou.append(i)
                # else:
                miou.extend(i)

                print("miou50:", np.mean(sum(miou) / len(miou)))
        print("miou50:", np.mean(sum(miou) / len(miou)))
        return np.mean(miou)

    def post_process(self, output, label):
        output_results = []
        label_results = []
        if type(output) == type({"s": 1}):
            output_results.append(output["boxes"])
            label_results.append(label["boxes"])
        else:
            for i in range(len(output)):
                output_results.append(output[i]["boxes"])
                label_results.append(label[i]["boxes"])
        output_result = torch.cat(output_results, dim=0)
        label_results = torch.cat(label_results, dim=0)
        ## formating to coco (xmin ymin w h)

        output_result[:, 2] = output_result[:, 2] - output_result[:, 0]
        output_result[:, 3] = output_result[:, 3] - output_result[:, 1]
        label_results[:, 0] = label_results[:, 0] - label_results[:, 2] / 2
        label_results[:, 1] = label_results[:, 1] - label_results[:, 3] / 2

        return output_result.unsqueeze(0), label_results.unsqueeze(0)

    def loading_data(self, data_dir="/home/parkjunsu/workspace/300/data"):
        # image_path = Path(data_dir / 'val_image')
        label_path = Path(data_dir) / "VL_KS_BBOX"
        json_paths = [f for f in label_path.rglob("*.json")]

        combined_data = {
            "image_id": [],
            "image_name": [],
            "image": [],
            "bboxes": [],
            "category": [],
        }

        i = 0
        for json_path in json_paths:

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            image_info = list(data.values())[0]

            filename = image_info["filename"]
            id = filename.split(".")[0]
            image_path = Path(data_dir) / "VS_KS" / filename
            combined_data["image_name"].append(id)
            combined_data["image_id"].append(i)
            i = i + 1

            annotations = []
            bboxes = []
            labels = []

            for region in image_info["regions"]:

                attrs = region["shape_attributes"]
                label = region["region_attributes"]["chi_id"]
                bbox = [attrs["x"], attrs["y"], attrs["width"], attrs["height"]]
                bboxes.append(bbox)
                labels.append(1)

            combined_data["bboxes"].append(bboxes)
            combined_data["category"].append(labels)
            combined_data["image"].append(str(image_path))
        dataset = Dataset.from_dict(combined_data)

        dataset = dataset.cast_column("image", Image())
        return dataset


if __name__ == "__main__":

    RT = RTWrapper()
    # dataset = RT.preprocess()
    # print(dataset[(0,1)])
    a = Path("/home/parkjunsu/workspace/300/data/VS_KS")
    results = RT.predict(
        [a / "K3_CHN_20130308050237_30.jpg", a / "K3_CHN_20130308050353_17.jpg"]
    )
    # RT.validate()
