#%%
import cv2
import matplotlib.pyplot as plt
import json
from glob import glob
import os
from tqdm import tqdm
# %%
def read_annotations(file_path):
    lines = open(file_path).readlines()
    data = []
    for l in lines:
        cls, x1, y1, x2, y2 = [int(i) for i in l.split()]
        data.append(
            {
                'box': [x1, y1, x2, y2],
                'class': cls
            }
        )
    return data

def path_to_id(path):
    return path.split('/')[-1].split('.')[0]

id_to_annotation = {
    path_to_id(p): read_annotations(p) 
    for p in glob("school_lunch/Annotations/*")}

for image_path in tqdm(glob("school_lunch/Images/*")):
    id_ = path_to_id(image_path)
    img = cv2.imread(image_path)
    
    for i, bbox in enumerate(id_to_annotation[id_]):
        x1, y1, x2, y2 = [max(0, i) for i in bbox['box']]
        object_image = img[y1:y2, x1:x2]
        cls = bbox['class']
        save_path = f'school_lunch/cropped/{cls}/'
        object_path = save_path + f'{id_}_{i}.jpg'
        if os.path.exists(object_path):
            break
        os.makedirs(save_path, exist_ok=True)
        cv2.imwrite(save_path + f'{id_}_{i}.jpg', object_image)
    



# %%
