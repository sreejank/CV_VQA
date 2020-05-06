import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as visionmodels
import numpy as np 

class VQAModel(nn.Module):
	def __init__(self,visualmodel='precomputed'):
		super().__init__()

		weights=torch.FloatTensor(np.load('data/embeddings_10.npy'))
		self.embedding_layer=nn.Embedding.from_pretrained(weights,freeze=False)
		self.lstm_layer=nn.LSTM(input_size=weights.shape[1],hidden_size=512,num_layers=2,batch_first=True)
		self.lstm_layer.bias_ih_l0.data.zero_()
		self.lstm_layer.bias_hh_l0.data.zero_()
		init.xavier_uniform(self.lstm_layer.weight_ih_l0)
		init.xavier_uniform(self.lstm_layer.weight_hh_l0) 

		if visualmodel=='precomputed':
			self.visual_module=nn.Linear(2048,512)
		else:
			resnet18=visionmodels.resnet18(pretrained=True)
			visual_layers=list(resnet18.children())[:-1]
			visual_layers.append(nn.Linear(2048,512))
			self.visual_module=nn.Sequential(visual_layers)

		self.fusion_module=nn.Linear(512,2)

	def forward(self,image,question): 
		embedding=self.embedding_layer(question) 
		lstm_out,_= self.lstm_layer(embedding)
		q=lstm_out.narrow(1,24,1).view(-1,512)
		v=self.visual_module(image)
		combined=torch.add(q,v)
		return F.log_softmax(self.fusion_module(combined),dim=1) 

class VQAModelAblated(nn.Module):
	def __init__(self,visualmodel='precomputed'):
		super().__init__()

		weights=torch.FloatTensor(np.load('data/embeddings_10.npy'))
		self.embedding_layer=nn.Embedding.from_pretrained(weights,freeze=False)

		if visualmodel=='precomputed':
			self.visual_module=nn.Linear(2048,300)
		else:
			resnet18=visionmodels.resnet18(pretrained=True)
			visual_layers=list(resnet18.children())[:-1]
			visual_layers.append(nn.Linear(2048,300))
			self.visual_module=nn.Sequential(visual_layers)

		self.fusion_module=nn.Linear(300,2)

	def forward(self,image,question): 
		embedding=self.embedding_layer(question) 
		q=torch.mean(embedding,dim=1)
		v=self.visual_module(image)
		combined=torch.add(q,v)
		return F.log_softmax(self.fusion_module(combined),dim=1) 











		