import json
import numpy as np 
import pickle
import nltk
from nltk.tokenize import word_tokenize
from torchnlp.word_to_vector import GloVe

nltk.download('punkt')

glove_cache=GloVe(cache='rawdata/glove')

train_questions=json.load(open('data/v2_OpenEnded_mscoco_train2014_questions.json',"r"))['questions']
val_questions=json.load(open('data/v2_OpenEnded_mscoco_val2014_questions.json',"r"))['questions']

train_question_dict={}
for question in train_questions:
	qid=question['question_id']
	train_question_dict[qid]=(question['image_id'],question['question'])

val_question_dict={}
for question in val_questions:
	qid=question['question_id']
	val_question_dict[qid]=(question['image_id'],question['question'])


train_img_features=pickle.load(open('data/train.pickle','rb'))
val_img_features=pickle.load(open('data/val.pickle','rb'))

train_answers=json.load(open("data/v2_mscoco_train2014_annotations.json","r"))['annotations']
val_answers=json.load(open("data/v2_mscoco_val2014_annotations.json","r"))['annotations']


answer_set={}
for answer in train_answers:
	for ans in answer['answers']:
		ans=ans['answer']
		if ans not in answer_set.keys():
			answer_set[ans]=0
		else:
			answer_set[ans]+=1

#common=sorted(list(answer_set.keys()),key=lambda k: answer_set[k],reverse=True)[:2]
common=['no','yes']
print(common)
common_array=np.asarray(common)
np.save('data/10_common.npy',common_array)
common_set=set(common)

max_seq_size=25

vocab=set()

train_data=[]
for answer in train_answers:
	common_idx=-1

	for ans in answer['answers']:
		ans=ans['answer']
		if ans in common_set:
			common_idx=np.where(common_array==ans)[0][0]
	if common_idx!=-1:
		qid=answer['question_id']
		imgid,question=train_question_dict[qid]
		question=word_tokenize(question.lower())[:-1]
		if len(question)>max_seq_size:
			print("Problem",len(question),question) 
		while len(question)<max_seq_size:
			question.append("<PAD>")
		resnet_vec=train_img_features[imgid]
		for word in question:
			if word not in vocab:
				vocab.add(word)
		train_data.append((np.asarray(question),resnet_vec,common_idx))
train_data=np.asarray(train_data)


val_data=[]
for answer in val_answers:
	common_idx=-1
	num_yes=0
	num_no=0
	for ans in answer['answers']:
		ans=ans['answer']
		if ans=='yes':
			num_yes+=1
		if ans=='no':
			num_no+=1

	if num_yes>0 or num_no>0:
		prevalent=np.argmax([num_no,num_yes])
		print(np.max([num_no,num_yes]),np.min([num_no,num_yes])) 
		qid=answer['question_id']
		imgid,question=val_question_dict[qid]
		question=word_tokenize(question.lower())[:-1]
		if len(question)>max_seq_size:
			print("Problem",len(question),question)
		while len(question)<max_seq_size:
			question.append("<PAD>")
		resnet_vec=val_img_features[imgid]
		for word in question:
			if word not in vocab: 
				vocab.add(word)
		val_data.append((np.asarray(question),resnet_vec,prevalent)) 
val_data=np.asarray(val_data)
 
vocab_set=list(vocab)
embedding_mat=np.zeros((len(vocab_set),300))
for i in range(embedding_mat.shape[0]):
	embedding_mat[i]=glove_cache[vocab_set[i]].numpy()
np.save('data/embeddings_10.npy',embedding_mat) 



vocab_converter={v:i for i,v in enumerate(list(vocab_set))}
for i in range(train_data.shape[0]):
	train_data[i][0]=np.asarray([vocab_converter[word] for word in train_data[i][0]])
for i in range(val_data.shape[0]):
	val_data[i][0]=np.asarray([vocab_converter[word] for word in val_data[i][0]])  

np.save('data/full_val_data_10.npy',val_data) 
np.save('data/full_train_data_10.npy',train_data) 








