
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



import warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

class Config:
    def __init__(self, device):
        # Model Architecture
        self.model_name = 'resnet50'
        self.image_embedding = 2048
        self.text_encoder_model = "distilbert-base-uncased"
        self.text_embedding = 768
        self.text_tokenizer = "distilbert-base-uncased"
        self.projection_dim = 256
        self.dropout = 0.1
        
        # Training Parameters
        self.batch_size = 16  # Reduced batch size
        self.num_workers = 2  # Reduced workers
        self.head_lr = 1e-3
        self.image_encoder_lr = 1e-4
        self.text_encoder_lr = 1e-5
        self.weight_decay = 1e-3
        self.patience = 5
        self.factor = 0.5
        self.epochs = 15
        self.temperature = 1.0
        self.max_length = 128  # Reduced sequence length
        self.gradient_accumulation_steps = 4  # Increased for smaller batches
        
        # Image Processing
        self.size = 160  # Reduced image size
        self.pretrained = True
        self.trainable = True
        
        # Memory Optimization
        self.gradient_checkpointing = True
        self.mixed_precision = True
        self.dataset_shard_size = 5000
        self.memory_efficient_loading = True
        
        # Distributed Training
        self.distributed = False
        self.world_size = 1
        self.rank = 0
        
        # Device
        self.device = device
        
        # Paths
        self.model_save_path = Path("models")
        self.log_dir = Path("logs")
        self.cache_dir = Path("cache")
        
        # Create necessary directories
        self.model_save_path.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)

class MemoryEfficientDataset(Dataset):
    def __init__(self, image_filenames, captions, tokenizer, transforms, image_path, cfg, shard_idx=None):
        self.cfg = cfg
        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.image_path = Path(image_path)
        self.transforms = transforms
        self.tokenizer = tokenizer
        
        # Implement sharding if specified
        if shard_idx is not None and cfg.dataset_shard_size > 0:
            start_idx = shard_idx * cfg.dataset_shard_size
            end_idx = min(start_idx + cfg.dataset_shard_size, len(image_filenames))
            self.image_filenames = image_filenames[start_idx:end_idx]
            self.captions = captions[start_idx:end_idx]
        
        # Validate images and create index mapping
        self._validate_and_create_mapping()
        
        # Tokenize captions efficiently
        self._tokenize_captions()

    def _validate_and_create_mapping(self):
        logging.info("Validating images and creating mapping...")
        self.valid_indices = []
        self.file_sizes = {}
        
        for idx, fname in enumerate(tqdm(self.image_filenames)):
            img_path = self.image_path / fname
            if self._validate_image(img_path):
                self.valid_indices.append(idx)
                self.file_sizes[idx] = img_path.stat().st_size
        
        logging.info(f"Found {len(self.valid_indices)} valid images")
        self._filter_valid_data()

    def _validate_image(self, path):
        try:
            return path.exists() and path.stat().st_size > 0
        except Exception:
            return False

    def _filter_valid_data(self):
        self.filtered_filenames = [self.image_filenames[i] for i in self.valid_indices]
        self.filtered_captions = [self.captions[i] for i in self.valid_indices]

    def _tokenize_captions(self):
        logging.info("Tokenizing captions...")
        self.encoded_captions = self.tokenizer(
            self.filtered_captions,
            padding='max_length',
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt"
        )

    def _load_image(self, img_path):
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                raise ValueError("Image could not be loaded")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logging.warning(f"Error loading image {img_path}: {e}")
            return None

    def __getitem__(self, idx):
        try:
            item = {
                key: self.encoded_captions[key][idx]
                for key in self.encoded_captions
            }
            
            img_path = self.image_path / self.filtered_filenames[idx]
            image = self._load_image(img_path)
            
            if image is None:
                # Return a different valid image if this one fails
                return self.__getitem__((idx + 1) % len(self))
            
            image = self.transforms(image=image)['image']
            item['image'] = torch.tensor(image).permute(2, 0, 1).float()
            item['caption'] = self.filtered_captions[idx]
            
            return item
        except Exception as e:
            logging.error(f"Error processing index {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))

    def __len__(self):
        return len(self.filtered_filenames)

class MemoryEfficientImageEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            num_classes=0,
            global_pool="avg"
        )
        
        # Enable gradient checkpointing if configured
        if cfg.gradient_checkpointing:
            self.model.set_grad_checkpointing(enable=True)
        
        for p in self.model.parameters():
            p.requires_grad = cfg.trainable

    def forward(self, x):
        return self.model(x)

class MemoryEfficientTextEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = DistilBertModel.from_pretrained(cfg.text_encoder_model)
        
        # Enable gradient checkpointing if configured
        if cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        for p in self.model.parameters():
            p.requires_grad = cfg.trainable
        
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]

class ProjectionHead(nn.Module):
    def __init__(self, cfg, embedding_dim):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, cfg.projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(cfg.projection_dim, cfg.projection_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.layer_norm = nn.LayerNorm(cfg.projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class OptimizedCLIPModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.image_encoder = MemoryEfficientImageEncoder(cfg)
        self.text_encoder = MemoryEfficientTextEncoder(cfg)
        self.image_projection = ProjectionHead(cfg, cfg.image_embedding)
        self.text_projection = ProjectionHead(cfg, cfg.text_embedding)
        self.temperature = cfg.temperature

    def forward(self, batch):
        def compute_image_features():
            return self.image_encoder(batch["image"])
            
        def compute_text_features():
            return self.text_encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
        
        # Use checkpointing if enabled
        if self.cfg.gradient_checkpointing:
            image_features = checkpoint(compute_image_features)
            text_features = checkpoint(compute_text_features)
        else:
            image_features = compute_image_features()
            text_features = compute_text_features()
        
        # Project features
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)
        
        # Calculate similarity and loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, 
            dim=-1
        )
        
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + texts_loss) / 2.0
        return loss.mean()

class MemoryTracker:
    @staticmethod
    def get_memory_usage():
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB
            gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**2  # MB
        else:
            gpu_memory = gpu_memory_reserved = 0
            
        ram_memory = psutil.Process().memory_info().rss / 1024**2  # MB
        return {
            'gpu_allocated': gpu_memory,
            'gpu_reserved': gpu_memory_reserved,
            'ram': ram_memory
        }

    @staticmethod
    def log_memory_usage(stage):
        memory = MemoryTracker.get_memory_usage()
        logging.info(f"Memory usage at {stage}:")
        logging.info(f"  GPU Memory Allocated: {memory['gpu_allocated']:.2f} MB")
        logging.info(f"  GPU Memory Reserved: {memory['gpu_reserved']:.2f} MB")
        logging.info(f"  RAM Usage: {memory['ram']:.2f} MB")

def get_transforms(mode="train", size=224):
    if mode == "train":
        return A.Compose([
            A.Resize(size, size, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(max_pixel_value=255.0, always_apply=True),
        ])
    else:
        return A.Compose([
            A.Resize(size, size, always_apply=True),
            A.Normalize(max_pixel_value=255.0, always_apply=True),
        ])

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def build_loaders(dataframe, tokenizer, cfg, image_path, mode="train"):
    transforms = get_transforms(mode=mode, size=cfg.size)
    
    # Calculate number of shards
    num_samples = len(dataframe)
    num_shards = math.ceil(num_samples / cfg.dataset_shard_size)
    
    # Create datasets for each shard
    datasets = []
    for shard_idx in range(num_shards):
        dataset = MemoryEfficientDataset(
            dataframe["image"].values,
            dataframe["caption"].values,
            tokenizer=tokenizer,
            transforms=transforms,
            image_path=image_path,
            cfg=cfg,
            shard_idx=shard_idx
        )
        datasets.append(dataset)
    
    # Combine datasets
    combined_dataset = torch.utils.data.ConcatDataset(datasets)
    
    # Create sampler for distributed training
    sampler = DistributedSampler(combined_dataset) if cfg.distributed else None
    
    # Create dataloader
    dataloader = DataLoader(
        combined_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=(mode == "train" and not cfg.distributed),
        pin_memory=True,
        drop_last=False,
        sampler=sampler
    )
    
    return dataloader

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()
    
    def reset(self):
        self.avg, self.sum, self.count = [0] * 3
    
    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count



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


#  use the model to predict the cloud cover of a new image
# load the model
model = pickle.load(open('models/catboost_model.sav', 'rb'))

# load the image
image = cv2.imread('dataset\data_images\Extracted Images\20160101075000.jpg')

# preprocess the image
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image = cv2.resize(image, (244, 244))
# image = np.moveaxis(image, -1, 0)

# predict the cloud cover
cloud_cover = model.predict(image)
print(cloud_cover)






