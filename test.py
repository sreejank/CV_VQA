from model import * 
import torch.utils.data as data
from torch.utils.data import Dataset
import numpy as np 
from torchvision import transforms
from torch.autograd import Variable
from train import VQADataset

if __name__=='__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model=VQAModelAblated().to(device)
	model.load_state_dict(torch.load('model_ablated.pt',map_location=device))
	dataloader=data.DataLoader(VQADataset('data/full_test_data_10.npy'),batch_size=64)
	model.eval()
	total_acc=[]
	for i,sample in enumerate(dataloader):
		image,question,answer=sample 
		v=image.to(device)
		q=question.to(device)
		a=answer.to(device) 
		probs=model(v,q)
		max_vals, max_indices = torch.max(probs,1)
		acc = (max_indices == a).sum().item()/max_indices.size()[0] 
		total_acc.append(acc)
	print(np.mean(total_acc))

