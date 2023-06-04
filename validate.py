#%%
from simple_train_shool_lunch import train_and_validate, predict_class
import cv2
from time import sleep
import matplotlib.pyplot as plt
# %%

photos_to_predict = [
    'school_lunch/cropped/10/021312000_5.jpg',
    'school_lunch/cropped/13/010455000_1.jpg',
    'school_lunch/cropped/3/010082000_0.jpg',
    'school_lunch/cropped/16/020267001_0.jpg',
]

valid_loss, train_loss, model, loader, predictions = train_and_validate()


#%%
predictions = [p[0][0] for p in predictions]
# %%
categories = [l.split(' ', 1) for l in open('school_lunch/category.txt').readlines()]
categories = {int(k): v.replace('\n', '') for k, v in categories}
categories
categories[13] = 'tofu'


# %%

def predict(path):
    cls_id = predict_class(model, model.embedding_size, loader, [path])[0][0][0]
    return categories[cls_id]
