from model import * 
import torch.utils.data as data
from torch.utils.data import Dataset
import numpy as np 
from torchvision import transforms
from torch.autograd import Variable
class VQADataset(Dataset):
	def __init__(self,data_path):
		data=np.load(data_path,allow_pickle=True)
		self.questions=data[:,0]
		self.questions=np.asarray([self.questions[i].astype('int64') for i in range(self.questions.shape[0])])
		self.images=data[:,1]
		self.images=np.asarray([self.images[i].astype('float32') for i in range(self.images.shape[0])])
		self.answers=data[:,2]
		self.answers=np.asarray([self.answers[i].astype('int64') for i in range(self.answers.shape[0])])
		self.n=self.questions.shape[0]
	def __len__(self):
		return self.n
	def __getitem__(self,idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		
		return (self.images[idx],self.questions[idx],self.answers[idx])  

if __name__=='__main__':
	print("Start")
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model=VQAModel().to(device)
	optimizer=torch.optim.Adam([param for param in model.parameters() if param.requires_grad])
	train_dataloader=data.DataLoader(VQADataset('data/full_train_data_10.npy'),batch_size=64)
	num_batches=len(train_dataloader)/64
	loss_function=nn.CrossEntropyLoss()
	for epoch in range(30): 
		model.train()
		total_loss=[]
		total_acc=[]
		batch_num=0
		for i,sample in enumerate(train_dataloader):
			image,question,answer=sample 
			v=image.to(device)
			q=question.to(device)
			a=answer.to(device) 
			optimizer.zero_grad()
			probs=model(v,q)

			loss=loss_function(probs,a) 
			loss.backward()
			optimizer.step() 
			total_loss.append(loss.item()) 
			max_vals, max_indices = torch.max(probs,1)
			train_acc = (max_indices == a).sum().item()/max_indices.size()[0] 
			total_acc.append(train_acc)
			#print(batch_num)
			batch_num+=1
			if i%10==0:
				print(i,"/",len(train_dataloader))
		print("Epoch ",epoch," average loss: ",np.mean(total_loss),np.mean(total_acc))
	torch.save(model.state_dict(), 'model.pt')


