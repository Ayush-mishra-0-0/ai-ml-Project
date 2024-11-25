import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import timm
from transformers import DistilBertModel

class MemoryEfficientImageEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            num_classes=0,
            global_pool="avg"
        )
        
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

class ClusterHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cluster_centers = nn.Parameter(
            torch.randn(cfg.num_clusters, cfg.projection_dim)
        )
        # Initialize cluster centers using Xavier initialization
        nn.init.xavier_uniform_(self.cluster_centers)

    def forward(self, x):
        # Calculate distances to cluster centers
        norm_squared = torch.sum((x.unsqueeze(1) - self.cluster_centers.unsqueeze(0)) ** 2, -1)
        # Convert distances to probabilities using Gaussian kernel
        similarities = torch.exp(-norm_squared / 2)
        # Normalize probabilities
        probabilities = similarities / similarities.sum(dim=1, keepdim=True)
        return probabilities

class CLOCModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.image_encoder = MemoryEfficientImageEncoder(cfg)
        self.text_encoder = MemoryEfficientTextEncoder(cfg)
        self.image_projection = ProjectionHead(cfg, cfg.image_embedding)
        self.text_projection = ProjectionHead(cfg, cfg.text_embedding)
        
        # Add cluster heads for both modalities
        self.image_cluster_head = ClusterHead(cfg)
        self.text_cluster_head = ClusterHead(cfg)
        
        self.temperature = cfg.temperature
        self.cluster_weight = cfg.cluster_weight  # Weight for clustering loss

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
        
        # Get cluster assignments
        image_clusters = self.image_cluster_head(image_embeddings)
        text_clusters = self.text_cluster_head(text_embeddings)
        
        # Calculate contrastive loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        
        # Calculate targets using both feature similarity and cluster assignments
        feature_targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, 
            dim=-1
        )
        
        # Calculate cluster-based targets
        cluster_targets = (image_clusters @ text_clusters.T)
        cluster_targets = cluster_targets / cluster_targets.sum(dim=-1, keepdim=True)
        
        # Combine feature and cluster targets
        targets = (1 - self.cluster_weight) * feature_targets + self.cluster_weight * cluster_targets
        
        # Calculate cross-modal losses
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        contrastive_loss = (images_loss + texts_loss) / 2.0
        
        # Calculate clustering consistency loss
        cluster_consistency_loss = F.mse_loss(image_clusters, text_clusters)
        
        # Combine losses
        total_loss = contrastive_loss.mean() + self.cfg.consistency_weight * cluster_consistency_loss
        
        return total_loss

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    return loss