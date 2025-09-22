from data_load import RTDataset,collate_fn
from pathlib import Path
import numpy as np
import torch
from transformers import AutoImageProcessor , AutoModelForObjectDetection
import albumentations as A
import os
import json
from datasets import Dataset,Image


class ModelWrapper:
    def __init__(self, method="nms", tta = None):
        self.model = self.load_model()
        assert method in ["nms", "soft-nms", "wbf"], "Invalid method"
        if method == "nms":
            self.ensemble_method = "nms"
        elif method == "soft-nms":
            self.ensemble_method = "soft_nms"
        else:
            self.ensemble_method = "wbf"
        self.tta = tta

    def preprocess(self, images, labels):
        # Implement any preprocessing steps if necessary
        return images, labels

    def load_model(self):
        raise NotImplementedError

    def predict(self, images):
        # final predicted outputs
        raise NotImplementedError

    def inference(self, images):
        # predicted outputs BEFORE NMS
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError




    def calculate_iou_matrix(self,bboxes1: torch.Tensor, bboxes2: torch.Tensor):
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
        
        return result_list

class RTWrapper(ModelWrapper):
    def __init__(self,method= "nms" , use_tta = True , checkpoint = '/workspace/RT_50/checkpoint-9744',device = 'cuda'):
        self.checkpoint = checkpoint
        self.device = device
        self.image_processor = AutoImageProcessor.from_pretrained(self.checkpoint)
        self.coco =  A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=1, min_width=1, min_height=1),
    )
        super().__init__(method,use_tta)

    def preprocess(self):
        val_data = self.loading_data()
        val_data = RTDataset(val_data, self.image_processor,transform=self.coco)
        return val_data
    def load_model(self):
        return AutoModelForObjectDetection.from_pretrained(self.checkpoint)
    def inference(self,image)-> dict:
        """
        arg: jpg image
        return: {'score' : tensor([ ~]) , 'boxes' : tensor([~])}
        """
        inputs = self.image_processor(image, return_tensors="pt")

        with torch.no_grad():
            output = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])

        result = self.image_processor.post_process_object_detection(output, threshold=0.25, target_sizes=target_sizes)
        return result
    def validate(self):
        n=2
        val_data = self.preprocess()
        val_data_loader = torch.utils.data.DataLoader(val_data , batch_size = n,collate_fn = collate_fn)
        miou = []
        
        with torch.no_grad():
            for batch in val_data_loader:
                
                label = batch['labels']
                output = self.model(batch['pixel_values'])
                target_size = torch.tensor([512,512]).unsqueeze(0).expand(n,-1)
                output = self.image_processor.post_process_object_detection(output,threshold = 0.25)
                print(output[0])
                output,label = self.post_process(output,label)
                
                print(output)
                print('--')
                print(label)
                i = self.calculate_iou_matrix(output,label)
               
                # if i.ndim == 0:
                #     miou.append(i)
                # else:
                miou.extend(i)
                
                print('miou:',i )
        print('miou50:',np.mean(miou))
        return np.mean(miou)

    def post_process(self,output,label): 
        output_results = []   
        label_results = []
        if type(output) == type({'s':1}):
            output_results.append(output['boxes'])
            label_results.append(label['boxes'])
        else:
            for i in range(len(output)):
                output_results.append(output[i]['boxes'])
                label_results.append(label[i]['boxes'])
        output_result = torch.cat(output_results , dim = 0)
        output_result[:,2] = output_result[:,2] - output_result[:,0]
        output_result[:,3] = output_result[:,3] - output_result[:,1]
        return output_result.unsqueeze(0) , torch.cat(label_results,dim = 0).unsqueeze(0)

    def loading_data(self,data_dir = '/workspace/data'):
        # image_path = Path(data_dir / 'val_image')
        label_path = Path(data_dir) / 'VL_KS_BBOX'
        json_paths = [f for f in label_path.rglob('*.json')]


        combined_data = {
            "image_id": [],
            "image_name":[],
            "image": [],
            "bboxes": [],
            "category": []
        }


        i = 0
        for json_path in json_paths:
            

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)


            image_info = list(data.values())[0]


            filename = image_info['filename']
            id = filename.split('.')[0]
            image_path = Path(data_dir) / 'VS_KS'/filename
            combined_data['image_name'].append(id)
            combined_data['image_id'].append(i)
            i = i+1

            annotations = []
            bboxes = []
            labels = []

            for region in image_info['regions']:

                attrs = region['shape_attributes']
                label = region['region_attributes']['chi_id']
                bbox = [attrs['x'], attrs['y'], attrs['width'], attrs['height']]
                bboxes.append(bbox)
                labels.append(1)


            combined_data["bboxes"].append(bboxes)
            combined_data['category'].append(labels)
            combined_data["image"].append(str(image_path))
        dataset = Dataset.from_dict(combined_data)


        dataset = dataset.cast_column("image", Image())
        return dataset
if __name__ == '__main__':
    RT = RTWrapper()
    # dataset = RT.preprocess()
    # print(dataset[(0,1)])
    RT.validate()