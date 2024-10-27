import os
import cv2
import gc
import itertools
import pickle
import numpy as np
import pandas as pd
import albumentations as A
import matplotlib.pyplot as plt
import torch
import timm
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.autonotebook import tqdm
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging

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
        self.batch_size = 32  # Reduced from 128 to handle large images
        self.num_workers = 4
        self.head_lr = 1e-3
        self.image_encoder_lr = 1e-4
        self.text_encoder_lr = 1e-5
        self.weight_decay = 1e-3
        self.patience = 3
        self.factor = 0.8
        self.epochs = 15
        self.temperature = 1.0
        self.max_length = 200
        
        # Image Processing
        self.size = 224
        self.pretrained = True
        self.trainable = True
        
        # Device
        self.device = device
        
        # Paths
        self.model_save_path = "models"
        self.log_dir = "logs"
        
        # Create necessary directories
        Path(self.model_save_path).mkdir(exist_ok=True)
        Path(self.log_dir).mkdir(exist_ok=True)

class WeatherCLIPDataset(Dataset):
    def __init__(self, image_filenames, captions, tokenizer, transforms, image_path, cfg):
        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.image_path = Path(image_path)
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=cfg.max_length
        )
        self.transforms = transforms
        
        # Validate images
        self.valid_indices = []
        self.image_mapping = {}
        
        logging.info("Validating weather imagery files...")
        for idx, fname in enumerate(tqdm(image_filenames)):
            img_path = self.image_path / fname
            if self._validate_weather_image(img_path):
                self.valid_indices.append(idx)
                self.image_mapping[idx] = img_path
        
        logging.info(f"Found {len(self.valid_indices)} valid images out of {len(image_filenames)}")
        
        # Filter data
        self._filter_valid_data()

    def _validate_weather_image(self, path):
        try:
            if not path.exists():
                return False
            img = cv2.imread(str(path))
            return img is not None
        except Exception as e:
            logging.warning(f"Error validating {path}: {str(e)}")
            return False

    def _filter_valid_data(self):
        self.captions = [self.captions[i] for i in self.valid_indices]
        self.image_filenames = [self.image_filenames[i] for i in self.valid_indices]
        self.encoded_captions = {
            key: [values[i] for i in self.valid_indices]
            for key, values in self.encoded_captions.items()
        }

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }
        
        try:
            image_path = self.image_mapping[idx]
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.transforms(image=image)['image']
            item['image'] = torch.tensor(image).permute(2, 0, 1).float()
            item['caption'] = self.captions[idx]
            return item
        except Exception as e:
            logging.error(f"Error processing {self.image_filenames[idx]}: {str(e)}")
            return self.__getitem__((idx + 1) % len(self))

    def __len__(self):
        return len(self.valid_indices)

class ImageEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = timm.create_model(
            cfg.model_name, 
            pretrained=cfg.pretrained, 
            num_classes=0, 
            global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = cfg.trainable

    def forward(self, x):
        return self.model(x)

class TextEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = DistilBertModel.from_pretrained(cfg.text_encoder_model)
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

class CLIPModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.image_encoder = ImageEncoder(cfg)
        self.text_encoder = TextEncoder(cfg)
        self.image_projection = ProjectionHead(cfg, cfg.image_embedding)
        self.text_projection = ProjectionHead(cfg, cfg.text_embedding)
        self.temperature = cfg.temperature

    def forward(self, batch):
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)
        
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
    dataset = WeatherCLIPDataset(
        dataframe["image"].values,
        dataframe["caption"].values,
        tokenizer=tokenizer,
        transforms=transforms,
        image_path=image_path,
        cfg=cfg
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=(mode == "train"),
        pin_memory=True if cfg.device.type == "cuda" else False,
        drop_last=False
    )
    return dataloader

def train_epoch(model, train_loader, optimizer, lr_scheduler, step, device):
    model.train()
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()
        
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=optimizer.param_groups[0]['lr'])
    return loss_meter

def valid_epoch(model, valid_loader, device):
    model.eval()
    loss_meter = AvgMeter()

    with torch.no_grad():
        for batch in tqdm(valid_loader):
            batch = {k: v.to(device) for k, v in batch.items() if k != "caption"}
            loss = model(batch)
            count = batch["image"].size(0)
            loss_meter.update(loss.item(), count)
    
    return loss_meter

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
    
    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"

def main():
    # Setup CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")

    # Initialize configuration
    cfg = Config(device)
    
    # Load and prepare data
    folder = 'dataset'
    img_folder = Path(folder) / 'data_images' / 'Extracted Images'
    
    logging.info("Loading data...")
    df = pd.read_csv(Path(folder) / "cloud_data_cleaned1.csv")
    df = df[['image_name', 'label', 'opaque_clouds']]
    df.columns = ['image', 'caption', 'cloudcover']
    
    # Train-Valid split
    train_df, valid_df = train_test_split(
        df, test_size=0.1, random_state=42
    )
    logging.info(f"Train size: {len(train_df)}, Valid size: {len(valid_df)}")
    
    # Initialize tokenizer and create data loaders
    tokenizer = DistilBertTokenizer.from_pretrained(cfg.text_tokenizer)
    train_loader = build_loaders(train_df, tokenizer, cfg, img_folder, mode="train")
    valid_loader = build_loaders(valid_df, tokenizer, cfg, img_folder, mode="valid")
    
    # Initialize model
    model = CLIPModel(cfg).to(device)
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    # Initialize optimizer and scheduler
    params = [
        {"params": model.module.image_encoder.parameters() if isinstance(model, nn.DataParallel) 
         else model.image_encoder.parameters(), "lr": cfg.image_encoder_lr},
        {"params": model.module.text_encoder.parameters() if isinstance(model, nn.DataParallel)
         else model.text_encoder.parameters(), "lr": cfg.text_encoder_lr},
        {"params": itertools.chain(
            model.module.image_projection.parameters() if isinstance(model, nn.DataParallel)
            else model.image_projection.parameters(),
            model.module.text_projection.parameters() if isinstance(model, nn.DataParallel)
            else model.text_projection.parameters()
        ), "lr": cfg.head_lr, "weight_decay": cfg.weight_decay}
    ]
    
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=cfg.patience, factor=cfg.factor
    )
    
    # Training loop
    best_loss = float('inf')
    best_epoch = 0
    
    logging.info("Starting training...")
    for epoch in range(cfg.epochs):
        logging.info(f"Epoch: {epoch + 1}/{cfg.epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, "epoch", device)
        valid_loss = valid_epoch(model, valid_loader, device)
        
        logging.info(f"Train Loss: {train_loss.avg:.4f}")
        logging.info(f"Valid Loss: {valid_loss.avg:.4f}")
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), "CLIP_model.pt")
            else:
                torch.save(model.state_dict(), "CLIP_model.pt")
            print("Saved Best Model!")
        
        lr_scheduler.step(valid_loss.avg)
        
        # Clear cache
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    # Save final model and configuration
    print("Saving final model and configuration...")
    with open('clip_mdl.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('clip_cfg.pkl', 'wb') as f:
        pickle.dump(cfg, f)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise