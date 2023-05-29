#%%
#%%
import torch
from torch import nn
import torch.nn.functional as F
import torchvision as tv
from PIL import Image
import numpy as np
from glob import glob
import random
from torch.utils.data import Dataset
from itertools import chain
import faiss
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from torch import optim
import os

#%%
device = 'mps'
#%%
import torchvision as tv
vgg16 = tv.models.vgg16(weights=tv.models.VGG16_Weights.DEFAULT)

vgg16 = vgg16.to(device)

transforms = tv.models.VGG16_Weights.DEFAULT.transforms()
#%%
train_transforms = tv.transforms.Compose([
        tv.transforms.Resize((224, 224)),
        tv.transforms.RandomHorizontalFlip(),
        #tv.transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
        tv.transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.2), shear=5, fill=0),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


#%%
class_image_path = glob('school_lunch/cropped/*/')
len(class_image_path)
#%%
def path_to_id(path):
    return path.split('/')[-2]

banned_classes = ['0', '20', ]
class_images = [glob(c + '*') for c in class_image_path if path_to_id(c) not in banned_classes]
total_samples = sum(map(len, class_images))
for c in class_images:
    random.shuffle(c)
    print(path_to_id(c[0]), len(c) / total_samples)
#%%

train_size = 0.8

train_class_images = [c[:int(train_size * len(c))] for c in class_images]
train_paths = list(chain(*train_class_images))
test_class_images = [c[int(train_size * len(c)):] for c in class_images]
test_paths = list(chain(*test_class_images))
for train_c, test_c in zip(train_class_images, test_class_images):
    print(len(train_c), len(train_c) + len(test_c))

class ImageData(Dataset):
    def __init__(self, paths, transforms, split="train") -> None:
        super().__init__()
        self.paths = paths
        self.split = split
        self.transforms = transforms
    
    def __getitem__(self, idx):
        path1 = self.paths[idx]
        cls1 = path_to_id(path1)
        img1 = self.transforms(Image.open(path1))
        return img1, cls1
    
    def get_path_of(self, idx):
        return self.paths[idx]
    
    def __len__(self):
        return len(self.paths)


train_data = ImageData(train_paths, train_transforms)
val_data = ImageData(test_paths, transforms, 'test')
batch_size = 72
train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset = val_data, batch_size=batch_size)

# %%
@torch.no_grad()
def get_features(model, X):
    X = model.features(X)
    X = F.avg_pool2d(X, kernel_size=X.shape[-2:])[..., 0, 0]
    return X
#%%
epoch_to_features = {}

vgg16.eval()
for e in range(50):
    features = []
    labels = []
    for images, cls in tqdm(train_loader):
        batch_features = get_features(vgg16, images.to(device)).cpu().numpy()
        features.append(batch_features)
        labels.extend(cls)
    features = np.concatenate(features)

    epoch_to_features[e] = {
        'features': features,
        'labels': np.array(labels, int)
    }
    print(e)

# %%
save_dir = 'school_lunch/preprocessed/vgg16/train/'
os.makedirs(save_dir, exist_ok=True)
for e, data in epoch_to_features.items():
    file = save_dir + f'{e}.npz'
    np.savez(file, **data)
    

# %%

vgg16.eval()
features = []
labels = []
for images, cls in tqdm(val_loader):
    batch_features = get_features(vgg16, images.to(device)).cpu().numpy()
    features.append(batch_features)
    labels.extend(cls)
features = np.concatenate(features)
labels = np.array(labels, int)
#%%
save_dir = 'school_lunch/preprocessed/vgg16/val/'
os.makedirs(save_dir, exist_ok=True)
file = save_dir + f'data.npz'
np.savez(file, **{'features': features, 'labels': labels})

# %%
train_data = ImageData(train_paths, transforms)

train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True)

vgg16.eval()
features = []
labels = []
for images, cls in tqdm(train_loader):
    batch_features = get_features(vgg16, images.to(device)).cpu().numpy()
    features.append(batch_features)
    labels.extend(cls)
features = np.concatenate(features)
labels = np.array(labels, int)
#%%
save_dir = 'school_lunch/preprocessed/vgg16/train/'
os.makedirs(save_dir, exist_ok=True)
file = save_dir + f'data.npz'
np.savez(file, **{'features': features, 'labels': labels})

# %%
