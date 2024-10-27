import os
import cv2
import gc
import itertools
import pickle
import numpy as np
import pandas as pd
import albumentations as A
import torch
import timm
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm.autonotebook import tqdm
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging
from torch.amp import autocast, GradScaler
import math
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import argparse
from torch.utils.checkpoint import checkpoint
import psutil
import json
from train import Config, OptimizedCLIPModel

folder = 'dataset'
df = pd.read_csv(folder+"/x.csv")
df = df[['image_name', 'label', 'opaque_clouds']]
df.columns = ['image', 'caption', 'cloudcover']
df.head()


x = df['image']
y = df['cloudcover']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=48)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.30, random_state=48)

print((x_train.shape, x_val.shape, x_test.shape))

CFG = Config("cuda")
model = OptimizedCLIPModel(CFG)

checkpoint = torch.load("models/best_model.pt", map_location=CFG.device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


# ----- Custom Dataset Loader ----- #
img_folder = "dataset/data_images/Extracted Images/"
# ----- Custom Dataset Loader ----- #
class SkyImage(Dataset):
	def __init__(self, img_dir, labels):
		self.img_dir = img_dir
		self.img_labels = labels
	def __len__(self):
		return len(self.img_dir)
	def __getitem__(self, idx):
		img_path = os.path.join(img_folder, self.img_dir[idx])
		#os.path.join("Extracted Images/", self.img_dir[idx])
		image = cv2.imread(img_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = cv2.resize(image, (244, 244))
		image = np.moveaxis(image, -1, 0)
		label = self.img_labels[idx]
		return image, label

# ----- Dataset ----- #
train_images = SkyImage(x_train.to_list(), y_train.to_list())
valid_images = SkyImage(x_val.to_list(), y_val.to_list())
test_images = SkyImage(x_test.to_list(), y_test.to_list())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# First move the model to the correct device
model = model.to(device)

def get_features(dataset):
    all_features, all_labels, all_embeddings = [], [], []
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=64)):
            # Move labels to device immediately when loading
            labels = labels.to(device)
            
            image_input = torch.tensor(np.stack(images), device=device, dtype=torch.float32)
            
            image_features = model.image_encoder(image_input)
            image_embeddings = model.image_projection(image_features)
            
            all_features.append(image_features)
            all_labels.append(labels)
            all_embeddings.append(image_embeddings)
    
    return (
        torch.cat(all_features),
        torch.cat(all_labels),
        torch.cat(all_embeddings)
    )

# Get features for each dataset
train_features, train_labels, train_embeddings = get_features(train_images)
valid_features, valid_labels, valid_embeddings = get_features(valid_images)
test_features, test_labels, test_embeddings = get_features(test_images)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

def evaluate(name, x, y, n, p): # p: features, #n: no of observations
	print("---------------------------------------------------")
	print("{} MAE: {}".format(name, mean_absolute_error(x, y)))
	print("{} RMSE: {}".format(name, mean_squared_error(x, y, squared=False)))
	print("{} MSE: {}".format(name, mean_squared_error(x, y)))
	r2 = r2_score(x, y)
	print("{} R2: {}".format(name, r2))
	print("---------------------------------------------------")

# Check if CUDA (GPU) is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# ----- Model Training ----- #

CB_model = CatBoostRegressor(iterations=700, learning_rate=0.1, max_depth=8, eval_metric='RMSE', random_seed=48)
CB_model.fit(train_features.cpu().numpy(), train_labels.cpu().numpy(),
			eval_set = (valid_features.cpu().numpy(), valid_labels.cpu().numpy()),
			use_best_model=True, plot=True, verbose=50)


# ----- Model Prediction ----- #

cbt_train_pred = CB_model.predict(train_features.cpu().numpy())
cbt_valid_pred = CB_model.predict(valid_features.cpu().numpy())
cbt_test_pred = CB_model.predict(test_features.cpu().numpy())


# ----- Model Evaluation ----- #

evaluate("Train", train_labels.cpu(), cbt_train_pred, len(cbt_train_pred), 1)
evaluate("Valid", valid_labels.cpu(), cbt_valid_pred, len(cbt_valid_pred), 1)
evaluate("Test", test_labels.cpu(), cbt_test_pred, len(cbt_test_pred), 1)

pickle.dump(CB_model, open('catboost_model.sav', 'wb'))