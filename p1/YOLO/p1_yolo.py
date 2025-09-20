from glob import glob
import json, os, yaml
from ultralytics import YOLO
from PIL import Image

path_valid_label = "/root/workspace/data_yolo/labels/val"
path_train_label = "/root/workspace/data_yolo/labels/train"
path_valid_image = "/root/workspace/data_yolo/images/val"
path_train_image = "/root/workspace/data_yolo/images/train"
weight_path = "/root/workspace/YOLOv11-Experiments/run1/weights/best.pt"

def preprocess_json(path):
  files = glob("*.json", root_dir=path)
  print(f"found {len(files)} files in {path}")
  for file in files:
      with open(os.path.join(path, file)) as f:
          data = json.load(f)
      item = list(data.values())[0]

      img_w = int(item['file_attributes']['img_width'])
      img_h = int(item['file_attributes']['img_height'])

      yolo_labels = []
      for obj in item['regions']:
          cls = int(obj['region_attributes']['chi_id'])-1
          x = obj['shape_attributes']['x']
          y = obj['shape_attributes']['y']
          w = obj['shape_attributes']['width']
          h = obj['shape_attributes']['height']
          cx = (x + w/2) / img_w
          cy = (y + h/2) / img_h
          nw = w / img_w
          nh = h / img_h
          yolo_labels.append(f"0 {cx} {cy} {nw} {nh}") # temporarily changed to 0

      out_file = file.replace(".json", ".txt")
      with open(os.path.join(path, out_file), "w") as f:
          f.write("\n".join(yolo_labels))

def dump_yaml():
# dump model yaml
    data_yaml = {
        'path':"/root/workspace/data_yolo/",
        'train' : 'images/train',
        'val': 'images/val',
        'nc' :1,
        'names': ['chimney']

    }
    with open("dataset.yaml", "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

# finetuning yolov11x
# execute ONLY when to train the model again!!!
def finetune_yolo():
    weight = weight_path if os.path.exists(weight_path) else 'yolo11x.pt'
    if weight == 'yolo11x.pt':
        print("pretrained weight not found, falling back to default yolo11x.pt")
    model = YOLO(weight)
    model.train(data = "dataset.yaml",
                epochs = 100,
                imgsz = 512,
                batch = 48,
                device = 0,
                lr0=0.0001,
                optimizer = 'AdamW',
                project = 'YOLOv11-Experiments',
                name = 'run2',
                exist_ok = True
                )
    return model


# Example prediction. Execute this cell multiple times to see different inference results.
def sample_prediction(model):
    i = 0
    files = glob("*.jpg", root_dir=path_valid_image)
    print(f"found {len(files)} files in {path_valid_image}")
    model.predict(source=(path_valid_image + '/' + files[i]), imgsz=512, save=True)
    image_path = "/content/runs/detect/predict2/" + files[i]
    im = Image.open(image_path)
    im.show()

if __name__ == "__main__":
    preprocess_json(path_valid_label)
    preprocess_json(path_train_label)
    dump_yaml()
    
    # load model
    # model = YOLO(weight_path)
    model = finetune_yolo()
    # sample_prediction(model)

    # validation
    results = model.val(data = "/content/dataset.yaml")
    # P 0.944, R 0.959, mAP50 0.986, mAP50-95 0.804
    result_tta = model.val(data = "/content/dataset.yaml", augment=True)
    # P 0.956, R 0.959, mAP 0.987, mAP50-95 0.804


