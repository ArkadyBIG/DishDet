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

valid_loss, train_loss, model, predictions = train_and_validate(predict_photos=photos_to_predict)
#%%
predictions = [p[0][0] for p in predictions]
# %%
categories = [l.split(' ', 1) for l in open('school_lunch/category.txt').readlines()]
categories = {int(k): v.replace('\n', '') for k, v in categories}
categories
#%%
categories[13] = 'tofu'
# %%
for i, (photo, predicted_class) in enumerate(zip(photos_to_predict, predictions)):
    img = cv2.imread(photo)
    img = cv2.resize(img, (300, 300))
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,30)
    fontScale              = 0.8
    fontColor              = (0,0,0)
    thickness              = 2
    lineType               = 2

    cv2.putText(img,
                categories[predicted_class], 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)
    
    plt.imshow(img[..., ::-1])
    plt.show()
    target = int(photo.split('cropped/')[-1].split('/')[0])
    print(f"Predicted label {categories[predicted_class]}")
    print(f"RealLabel label {categories[target]}")
    sleep(3)

    

# %%
