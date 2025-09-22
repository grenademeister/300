import json
import os
from datasets import Dataset, Image
import numpy as np
import torch
def loading_data(label_path  , jpg_path):
    all_files = os.listdir(label_path)
    json_files = [f for f in all_files if f.endswith('.json')]


    combined_data = {
        "image_id": [],
        "image_name":[],
        "image": [],
        "bboxes": [],
        "category": []
    }
    error_bbox = 0
    i = 0
    for json_file in json_files:
        json_path = os.path.join(label_path, json_file)

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        image_info = list(data.values())[0]
        filename = image_info['filename']
        id = filename.split('.')[0]
        image_path = os.path.join(jpg_path, filename)
        combined_data['image_name'].append(id)
        combined_data['image_id'].append(i)
        i = i+1
        annotations = []
        bboxes = []
        labels = []
        for region in image_info['regions']:

            attrs = region['shape_attributes']
            label = region['region_attributes']['chi_id']
            bbox = (attrs['x'], attrs['y'], attrs['width'], attrs['height'])
            bboxes.append(bbox)
            print(bbox)
            labels.append(0)
        combined_data["bboxes"].append(bboxes)
        combined_data['category'].append(labels)
        combined_data["image"].append(image_path)
    # if len(bboxes) !=4:
    #     error_bbox +=1
        
    #     print('-----error:',json_path)
    #     print(bboxes)
    # else:
    #     print(bboxes)
        # print(bboxes)
   
    dataset = Dataset.from_dict(combined_data)
    dataset = dataset.cast_column("image", Image())
    
    return dataset

class RTDataset(Dataset):
    def __init__(self, dataset, image_processor, transform=None):
        self.dataset = dataset
        self.image_processor = image_processor
        self.transform = transform

    @staticmethod
    def format_image_annotations_as_coco(image_id, categories, boxes):
        """Format one set of image annotations to the COCO format

        Args:
            image_id (str): image id. e.g. "0001"
            categories (List[int]): list of categories/class labels corresponding to provided bounding boxes
            boxes (List[Tuple[float]]): list of bounding boxes provided in COCO format
                ([center_x, center_y, width, height] in absolute coordinates)

        Returns:
            dict: {
                "image_id": image id,
                "annotations": list of formatted annotations
            }
        """
        annotations = []
        for category, bbox in zip(categories, boxes):
            formatted_annotation = {
                "image_id": image_id,
                "category_id": category,
                "bbox": bbox,
                "iscrowd": 0,
                "area": bbox[2] * bbox[3],
            }
         
            annotations.append(formatted_annotation)

        return {
            "image_id": image_id,
            "annotations": annotations,
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        sample = self.dataset[idx]
     
        image_id = sample["image_id"]
        image = sample["image"]
        # print(image)
        # print(type(image))
        boxes = sample["bboxes"]
        categories = sample["category"]
     
        # Convert image to RGB numpy array
        image = np.array(image)
     
        formatted_annotations = []
        # Apply augmentations
        if self.transform:
            if type(idx) == type(1):
                transformed = self.transform(image=image, bboxes=boxes, category=categories)
                image = transformed["image"]
                boxes = transformed["bboxes"]
                categories = transformed["category"]
                formatted_annotations = self.format_image_annotations_as_coco(image_id, categories, boxes)
            else:
                for obj_index in range(len(image)):
                    transformed = self.transform(image=image[obj_index], bboxes=boxes[obj_index], category=categories[obj_index])
                    image[obj_index] = transformed["image"]
                    boxes[obj_index] = transformed["bboxes"]
                    categories[obj_index] = transformed["category"]
                    formatted_annotations.append(self.format_image_annotations_as_coco(image_id[obj_index], categories[obj_index], boxes[obj_index]))
        else:
            if type(idx) == type(1):
                formatted_annotations.append(self.format_image_annotations_as_coco(image_id,categories,boxes))
            else:
                for obj_index in range(len(image)):
                    formatted_annotations.append(self.format_image_annotations_as_coco(image_id[obj_index],categories[obj_index],boxes[obj_index]))

        # Format annotations in COCO format for image_processor
        
        
        # Apply the image processor transformations: resizing, rescaling, normalization
        result = self.image_processor(
            images=image, annotations=formatted_annotations, return_tensors="pt"
        )
      
      
        
        # Image processor expands batch dimension, lets squeeze it
        # result = {k: v[0] for k, v in result.items()}

        return result



def collate_fn(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    
    return data