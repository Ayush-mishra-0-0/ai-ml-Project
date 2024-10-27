_K='gpu_reserved'
_J='gpu_allocated'
_I='cpu'
_H=False
_G='cuda'
_F='none'
_E='train'
_D='caption'
_C=None
_B='image'
_A=True
import os,cv2,gc,itertools,pickle,numpy as np,pandas as pd,albumentations as A,torch,timm
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm.autonotebook import tqdm
from transformers import DistilBertModel,DistilBertTokenizer
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging
from torch.amp import autocast,GradScaler
import math
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist,argparse
from torch.utils.checkpoint import checkpoint
import psutil,json,warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',handlers=[logging.FileHandler('training.log'),logging.StreamHandler()])
class Config:
	def __init__(A,device):B='distilbert-base-uncased';A.model_name='resnet50';A.image_embedding=2048;A.text_encoder_model=B;A.text_embedding=768;A.text_tokenizer=B;A.projection_dim=256;A.dropout=.1;A.batch_size=16;A.num_workers=2;A.head_lr=.001;A.image_encoder_lr=.0001;A.text_encoder_lr=1e-05;A.weight_decay=.001;A.patience=5;A.factor=.5;A.epochs=15;A.temperature=1.;A.max_length=128;A.gradient_accumulation_steps=4;A.size=160;A.pretrained=_A;A.trainable=_A;A.gradient_checkpointing=_A;A.mixed_precision=_A;A.dataset_shard_size=5000;A.memory_efficient_loading=_A;A.distributed=_H;A.world_size=1;A.rank=0;A.device=device;A.model_save_path=Path('models');A.log_dir=Path('logs');A.cache_dir=Path('cache');A.model_save_path.mkdir(exist_ok=_A);A.log_dir.mkdir(exist_ok=_A);A.cache_dir.mkdir(exist_ok=_A)
class MemoryEfficientDataset(Dataset):
	def __init__(A,image_filenames,captions,tokenizer,transforms,image_path,cfg,shard_idx=_C):
		F=shard_idx;E=captions;C=image_filenames;B=cfg;A.cfg=B;A.image_filenames=C;A.captions=list(E);A.image_path=Path(image_path);A.transforms=transforms;A.tokenizer=tokenizer
		if F is not _C and B.dataset_shard_size>0:D=F*B.dataset_shard_size;G=min(D+B.dataset_shard_size,len(C));A.image_filenames=C[D:G];A.captions=E[D:G]
		A._validate_and_create_mapping();A._tokenize_captions()
	def _validate_and_create_mapping(A):
		logging.info('Validating images and creating mapping...');A.valid_indices=[];A.file_sizes={}
		for(B,D)in enumerate(tqdm(A.image_filenames)):
			C=A.image_path/D
			if A._validate_image(C):A.valid_indices.append(B);A.file_sizes[B]=C.stat().st_size
		logging.info(f"Found {len(A.valid_indices)} valid images");A._filter_valid_data()
	def _validate_image(A,path):
		try:return path.exists()and path.stat().st_size>0
		except Exception:return _H
	def _filter_valid_data(A):A.filtered_filenames=[A.image_filenames[B]for B in A.valid_indices];A.filtered_captions=[A.captions[B]for B in A.valid_indices]
	def _tokenize_captions(A):logging.info('Tokenizing captions...');A.encoded_captions=A.tokenizer(A.filtered_captions,padding='max_length',truncation=_A,max_length=A.cfg.max_length,return_tensors='pt')
	def _load_image(D,img_path):
		A=img_path
		try:
			B=cv2.imread(str(A))
			if B is _C:raise ValueError('Image could not be loaded')
			return cv2.cvtColor(B,cv2.COLOR_BGR2RGB)
		except Exception as C:logging.warning(f"Error loading image {A}: {C}");return
	def __getitem__(A,idx):
		B=idx
		try:
			D={C:A.encoded_captions[C][B]for C in A.encoded_captions};E=A.image_path/A.filtered_filenames[B];C=A._load_image(E)
			if C is _C:return A.__getitem__((B+1)%len(A))
			C=A.transforms(image=C)[_B];D[_B]=torch.tensor(C).permute(2,0,1).float();D[_D]=A.filtered_captions[B];return D
		except Exception as F:logging.error(f"Error processing index {B}: {F}");return A.__getitem__((B+1)%len(A))
	def __len__(A):return len(A.filtered_filenames)
class MemoryEfficientImageEncoder(nn.Module):
	def __init__(B,cfg):
		A=cfg;super().__init__();B.model=timm.create_model(A.model_name,pretrained=A.pretrained,num_classes=0,global_pool='avg')
		if A.gradient_checkpointing:B.model.set_grad_checkpointing(enable=_A)
		for C in B.model.parameters():C.requires_grad=A.trainable
	def forward(A,x):return A.model(x)
class MemoryEfficientTextEncoder(nn.Module):
	def __init__(A,cfg):
		B=cfg;super().__init__();A.model=DistilBertModel.from_pretrained(B.text_encoder_model)
		if B.gradient_checkpointing:A.model.gradient_checkpointing_enable()
		for C in A.model.parameters():C.requires_grad=B.trainable
		A.target_token_idx=0
	def forward(A,input_ids,attention_mask):B=A.model(input_ids=input_ids,attention_mask=attention_mask);C=B.last_hidden_state;return C[:,A.target_token_idx,:]
class ProjectionHead(nn.Module):
	def __init__(A,cfg,embedding_dim):B=cfg;super().__init__();A.projection=nn.Linear(embedding_dim,B.projection_dim);A.gelu=nn.GELU();A.fc=nn.Linear(B.projection_dim,B.projection_dim);A.dropout=nn.Dropout(B.dropout);A.layer_norm=nn.LayerNorm(B.projection_dim)
	def forward(A,x):B=A.projection(x);x=A.gelu(B);x=A.fc(x);x=A.dropout(x);x=x+B;x=A.layer_norm(x);return x
class OptimizedCLIPModel(nn.Module):
	def __init__(B,cfg):A=cfg;super().__init__();B.cfg=A;B.image_encoder=MemoryEfficientImageEncoder(A);B.text_encoder=MemoryEfficientTextEncoder(A);B.image_projection=ProjectionHead(A,A.image_embedding);B.text_projection=ProjectionHead(A,A.text_embedding);B.temperature=A.temperature
	def forward(A,batch):
		B=batch
		def E():return A.image_encoder(B[_B])
		def G():return A.text_encoder(input_ids=B['input_ids'],attention_mask=B['attention_mask'])
		if A.cfg.gradient_checkpointing:H=checkpoint(E);I=checkpoint(G)
		else:H=E();I=G()
		C=A.image_projection(H);D=A.text_projection(I);J=D@C.T/A.temperature;L=C@C.T;M=D@D.T;K=F.softmax((L+M)/2*A.temperature,dim=-1);N=cross_entropy(J,K,reduction=_F);O=cross_entropy(J.T,K.T,reduction=_F);P=(O+N)/2.;return P.mean()
class MemoryTracker:
	@staticmethod
	def get_memory_usage():
		if torch.cuda.is_available():A=torch.cuda.memory_allocated()/1024**2;B=torch.cuda.memory_reserved()/1024**2
		else:A=B=0
		C=psutil.Process().memory_info().rss/1024**2;return{_J:A,_K:B,'ram':C}
	@staticmethod
	def log_memory_usage(stage):A=MemoryTracker.get_memory_usage();logging.info(f"Memory usage at {stage}:");logging.info(f"  GPU Memory Allocated: {A[_J]:.2f} MB");logging.info(f"  GPU Memory Reserved: {A[_K]:.2f} MB");logging.info(f"  RAM Usage: {A['ram']:.2f} MB")
def get_transforms(mode=_E,size=224):
	B=size
	if mode==_E:return A.Compose([A.Resize(B,B,always_apply=_A),A.HorizontalFlip(p=.5),A.RandomBrightnessContrast(p=.5),A.Normalize(max_pixel_value=255.,always_apply=_A)])
	else:return A.Compose([A.Resize(B,B,always_apply=_A),A.Normalize(max_pixel_value=255.,always_apply=_A)])
def cross_entropy(preds,targets,reduction=_F):
	A=reduction;C=nn.LogSoftmax(dim=-1);B=(-targets*C(preds)).sum(1)
	if A==_F:return B
	elif A=='mean':return B.mean()
def build_loaders(dataframe,tokenizer,cfg,image_path,mode=_E):
	B=dataframe;A=cfg;E=get_transforms(mode=mode,size=A.size);F=len(B);G=math.ceil(F/A.dataset_shard_size);C=[]
	for H in range(G):I=MemoryEfficientDataset(B[_B].values,B[_D].values,tokenizer=tokenizer,transforms=E,image_path=image_path,cfg=A,shard_idx=H);C.append(I)
	D=torch.utils.data.ConcatDataset(C);J=DistributedSampler(D)if A.distributed else _C;K=DataLoader(D,batch_size=A.batch_size,num_workers=A.num_workers,shuffle=mode==_E and not A.distributed,pin_memory=_A,drop_last=_H,sampler=J);return K
class AvgMeter:
	def __init__(A,name='Metric'):A.name=name;A.reset()
	def reset(A):A.avg,A.sum,A.count=[0]*3
	def update(A,val,count=1):B=count;A.count+=B;A.sum+=val*B;A.avg=A.sum/A.count
def train_epoch(model,train_loader,optimizer,scheduler,scaler,device,cfg):
	G=scheduler;F=model;E=scaler;D=optimizer;A=cfg;F.train();H=AvgMeter();J=MemoryTracker();D.zero_grad()
	for(I,B)in enumerate(tqdm(train_loader)):
		J.log_memory_usage(f"Batch {I} start");B={A:B.to(device)for(A,B)in B.items()if A!=_D}
		with autocast(device_type=_G if torch.cuda.is_available()else _I,enabled=A.mixed_precision):C=F(B);C=C/A.gradient_accumulation_steps
		E.scale(C).backward()
		if(I+1)%A.gradient_accumulation_steps==0:
			E.step(D);E.update();D.zero_grad()
			if G is not _C:G.step()
		K=B[_B].size(0);H.update(C.item()*A.gradient_accumulation_steps,K)
	return H
def validate_epoch(model,valid_loader,device):
	B=model;B.eval();D=MemoryTracker();C=AvgMeter()
	with torch.no_grad():
		for A in tqdm(valid_loader):
			D.log_memory_usage(f"Batch start");A={A:B.to(device)for(A,B)in A.items()if A!=_D}
			with autocast(device_type=_G if torch.cuda.is_available()else _I,enabled=_A):E=B(A)
			F=A[_B].size(0);C.update(E.item(),F)
	return C
def main():
	S='LayerNorm.weight';R='bias';Q='weight_decay';P='params';C=torch.device(_G if torch.cuda.is_available()else _I);T=GradScaler(enabled=_A);A=Config(C);J=Path('dataset');K=J/'data_images'/'Extracted Images';logging.info('Loading and preparing data...');D=pd.read_csv(J/'cloud_data_cleaned1.csv');D=D[['image_name','label','opaque_clouds']];D.columns=[_B,_D,'cloudcover'];L,M=train_test_split(D,test_size=.1,random_state=42);logging.info(f"Train size: {len(L)}, Valid size: {len(M)}");N=DistilBertTokenizer.from_pretrained(A.text_tokenizer);U=build_loaders(L,N,A,K,mode=_E);V=build_loaders(M,N,A,K,mode='valid');B=OptimizedCLIPModel(A)
	if torch.cuda.device_count()>1:B=nn.DataParallel(B)
	B=B.to(C);W=[{P:[B for(A,B)in B.named_parameters()if not any(B in A for B in[R,S])],Q:A.weight_decay},{P:[B for(A,B)in B.named_parameters()if any(B in A for B in[R,S])],Q:.0}];G=torch.optim.AdamW(W);H=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(G,T_0=A.epochs,T_mult=1,eta_min=1e-06);E=float('inf')
	for O in range(A.epochs):
		logging.info(f"Epoch: {O+1}/{A.epochs}");X=train_epoch(B,U,G,H,T,C,A);logging.info(f"Train Loss: {X.avg:.4f}");F=validate_epoch(B,V,C);logging.info(f"Valid Loss: {F.avg:.4f}")
		if F.avg<E:E=F.avg;Y={'epoch':O,'model_state_dict':B.state_dict(),'optimizer_state_dict':G.state_dict(),'scheduler_state_dict':H.state_dict(),'loss':E,'config':A};torch.save(Y,A.model_save_path/'best_model.pt');logging.info(f"Saved best model with loss: {E:.4f}")
		H.step(F.avg)
		if C.type==_G:torch.cuda.empty_cache()
	logging.info('Training complete!');print('Saving final model and configuration...')
	with open('clip_mdl.pkl','wb')as I:pickle.dump(B,I)
	with open('clip_cfg.pkl','wb')as I:pickle.dump(A,I)
if __name__=='__main__':
	try:main()
	except Exception as e:print(f"An error occurred: {str(e)}");raise