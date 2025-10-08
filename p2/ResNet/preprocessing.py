import json


def load_json(json_path:str)->dict:
    with open(json_path, "r" , encoding="utf-8") as file:
        data = json.load(file)
    return data

def extract_bbox(json:dict)->tuple[list,list]:
    """
    Arg:
        json: law json dict data
    returns:
        ([[xmin,xmax,ymin,ymax]] , [int])
    
    """
    inner = list(json.values())[0]['regions']
    bbox = []
    height =[]
    for r in inner:
        all_x = r['shape_attributes']['all_points_x']
        all_y = r['shape_attributes']['all_points_y']
        bbox.append([all_x[0],all_x[1],all_y[0],all_y[1]])
        height.append(r['region_attributes']['chi_height_m'])
    return(bbox ,height )




if __name__=="__main__":
  path = '/home/parkjunsu/workspace/300/data/val_p2/K3A_CHN_20240424055006_8.json'
  data = load_json(path)
  print(extract_bbox(data))