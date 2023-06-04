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
from itertools import chain, cycle, repeat
import faiss                            
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from torch import optim
import os

#%%
device = 'mps'


class ImageData(Dataset):
    def __init__(self, path, split="train") -> None:
        super().__init__()
        self.path = path
        self.split = split    # train or valid
        self.epoch = 0
        self.features = np.array(0)
        self.labels = np.array(0)
        self.load_epoch(self.epoch)
        
        
    
    def load_epoch(self, value):
        if self.split != 'train':
            data = np.load(self.path + f'data.npz')
             
        else:
            data = np.load(self.path + f'{value}.npz')
            self.epoch = value
        self.features = data['features']
        self.labels = data['labels']
    
    
    def __getitem__(self, idx):
        features1 = self.features[idx]
        cls1 = self.labels[idx]
        cls2 = cls1
        while True:
            idx2 = random.randint(0, len(self.labels) - 1)
            if cls2 == self.labels[idx2] and idx2 != idx:
                features2 = self.features[idx2]
                break
        
        while True:
            idx3 = random.randint(0, len(self.labels) - 1)
            cls3 = self.labels[idx3]
            if cls3 != cls1:
                features3 = self.features[idx3]
                break      

        return features1, features2, features3, cls1, cls2, cls3
    
    def iter_all(self):
        for i in range(len(self)):
            yield torch.Tensor(self.features[i]), self.labels[i]
    
    def __len__(self):
        return len(self.labels)
    
class SimpleDataset(Dataset):
    def __init__(self, x, y) -> None:
        super().__init__()
        self.x = x
        self.y = y
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def __len__(self):
        return len(self.x)



train_paths = 'school_lunch/preprocessed/vgg16/train/'
test_paths = 'school_lunch/preprocessed/vgg16/val/'

class Model(nn.Module):
    def __init__(self, linears, features=None, ) -> None:
        super().__init__()
        if features is not None:
            for p in features.parameters():
                p.requires_grad = False
            self.features = features
        
        self.linears = nn.ModuleList(
            [nn.Linear(linears[i], linears[i + 1]) for i in range(len(linears) - 1)]
        )
            
        self.activation = nn.Sigmoid()
        
        self.embedding_size = linears[-1]
    
    def get_features(self, X):
        X = self.features(X)
        X = F.avg_pool2d(X, kernel_size=X.shape[-2:])[..., 0, 0]
        return X
    
    def forward_from_features(self, X):
        for i, l in enumerate(self.linears):
            X = l(X)
            X = self.activation(X)
        X = (X - 0.5) * 2
        X = F.normalize(X)
        return X
    
    def forward(self, X):
        X = self.get_features(X)
        X = self.dropout(X)
        X = self.forward_from_features(X)
        return X

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return torch.sqrt((x1 - x2).pow(2).sum(1))

    def forward(self, anchor, positive, negative):
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.clamp(distance_positive - distance_negative + self.margin, min=0)
        return losses.mean()
@torch.no_grad()
def load_indexer(model, embedding_size, feature_loader):
    model.eval()
    index = faiss.IndexFlatL2(embedding_size)
    train_lables = []
    for features, cls in (feature_loader):
        emb = model.forward_from_features(features.to(device))
        index.add(emb.cpu().numpy())
        train_lables.extend(cls.numpy().tolist())
    return index, train_lables

@torch.no_grad()
def get_accuracy(model, embedding_size, feature_loader, val_features_loader):
    model.eval()
    index, train_lables = load_indexer(model, embedding_size, feature_loader)
    result = []
    for features, cls in (val_features_loader):
        emb = model.forward_from_features(features.to(device))
        _, I = index.search(emb.cpu().numpy(), 1)
        indecies = np.take(train_lables, I)
        for c, target in zip(cls, indecies):
            result.append(bool(c == target[0]))
    return np.mean(result)

vgg16 = tv.models.vgg16(weights=tv.models.VGG16_Weights.DEFAULT)
vgg16 = vgg16.to(device)
transforms = tv.models.VGG16_Weights.DEFAULT.transforms()

@torch.no_grad()
def get_features(model, path_or_image):
    if isinstance(path_or_image, str):
        path_or_image = Image.open(Image.open(path))
    X = transforms(path_or_image)
    X = model.features(X.to(device))
    X = F.avg_pool2d(X, kernel_size=X.shape[-2:])[..., 0, 0]
    return X

@torch.no_grad()
def predict_class(model, embedding_size, feature_loader, photo_paths: list):
    model.eval()
    index, train_lables = load_indexer(model, embedding_size, feature_loader)
    result = []
    for path in photo_paths:
        emb = model.forward_from_features(get_features(vgg16, path)[None])
        _, I = index.search(emb.cpu().numpy(), 1)
        indecies = np.take(train_lables, I)
        result.append(indecies)
    return result

def train_and_validate(embedding_size=16, predict_photos=[]):
    
    train_data = ImageData(train_paths)
    features_data = ImageData(train_paths, 'test')
    features_data = SimpleDataset(features_data.features.copy(), features_data.labels.copy())

    val_features_data = ImageData(test_paths, 'test')
    val_features_data = SimpleDataset(val_features_data.features.copy(), val_features_data.labels.copy())
    val_data = ImageData(test_paths, 'test')
    batch_size = 72

    train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset = val_data, batch_size=batch_size, shuffle=False)
    feature_loader = torch.utils.data.DataLoader(dataset = features_data, batch_size=batch_size, shuffle=False)
    val_features_loader = torch.utils.data.DataLoader(dataset = val_features_data, batch_size=batch_size, shuffle=False)

    model = Model(linears=[512, 256, embedding_size])
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, )
    triplet_loss = TripletLoss(0.5)
    epochs = 1
    train_loss, valid_loss = [], []
    for epoch_i in range(epochs):
        train_loader.dataset.load_epoch(epoch_i)
        model.train()
        epoch_loss = 0
        for i, data in enumerate((train_loader)):
            optimizer.zero_grad()
            x1,x2,x3, *_ = data
            e1 = model.forward_from_features(x1.to(device))
            e2 = model.forward_from_features(x2.to(device))
            e3 = model.forward_from_features(x3.to(device))
            loss = triplet_loss(e1,e2,e3)
            
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        model.eval()
        accuracy = get_accuracy(
            model,
            embedding_size=embedding_size, 
            feature_loader=feature_loader, 
            val_features_loader=val_features_loader)
        with torch.no_grad():
            val_loss = 0
            for data in (val_loader):
                x1,x2,x3, *_ = data
                e1 = model.forward_from_features(x1.to(device))
                e2 = model.forward_from_features(x2.to(device))
                e3 = model.forward_from_features(x3.to(device))

                loss = triplet_loss(e1,e2,e3)
                val_loss += loss.item()
            val_loss = val_loss / len(val_loader)
            epoch_loss = epoch_loss / len(train_loader)
        valid_loss.append(val_loss)
        train_loss.append(epoch_loss)
        print(f"{epoch_i}: Train Loss: {epoch_loss} Val loss: {val_loss} Acc: {accuracy}")
    clss = predict_class(model, embedding_size, feature_loader, predict_photos)
    return valid_loss, train_loss, model, feature_loader, clss



