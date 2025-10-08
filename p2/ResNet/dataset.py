import torch
from torch.utils.data import Dataset
from pathlib import Path
from preprocessing import load_json,extract_bbox
from PIL import Image
class ResDataset(Dataset):
    def __init__(self, img_dir , label_dir,preprocessor,device,transformer = None):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.preprocessor =preprocessor
        self.transformer = transformer
        self.device = device
        self.datas =[] ##(img_path ,  bbox,height)
        for file in self.label_dir.glob('*.json'):
            json = load_json(file)
            bboxes,heights = extract_bbox(json)
            for bbox,height in zip(bboxes,heights):
                self.datas.append((file.with_suffix('.jpg'),bbox,height))
        print('dataset:',len(self.datas))
    def __len(self):
        return len(self.datas)
    def __getitem__(self,index)->dict[str,torch.tensor]:
        """
        Arg:
            index
        Returns:
            {'image':tensor[512,512] ,'bbox':tensor[4],'height':tensor[1] }
          
        """
        data = self.datas[index]
        
        img = Image.open(data[0])
        img = self.preprocessor(img,return_tensors = 'pt',device = self.device).to(self.device)
        bbox = torch.tensor(data[1],dtype=torch.float32).to(self.device)
        height = torch.tensor(data[2],dtype=torch.float32).to(self.device)

        return {
            'image':img,
            'bbox': bbox,
            'height':height
        }
      