import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import matplotlib.pyplot as plt
import pickle
import sqlite3
import sys
from torch.autograd import Variable
from random import shuffle
from tqdm import tqdm

import model


def check_accuracy(net, dev_path):
	dataint = open(dev_path, 'rb')
	pos, neg, tie = 0,0,0
	while dataint:
		try:
	
			pronoun_data = pickle.load(dataint)
			ref_sen_len = len(pronoun_data['reference_sentence'])
			ref_con_len = len(pronoun_data['reference_context'])
			if ref_con_len - ref_sen_len <=0: #skip if there is no context; can comment out 
				continue

			pos_context_input = [pronoun_data['reference_context']]
			pos_sentence_input = [] #make context input blank and use this if running a no context version
			##Getting pronouns only from the last sentence but adjusting indices for the full context
			pos_context_pids = get_pronoun_idx(pronoun_data['reference_context'], pronoun_list)
			pos_sentenceonly_pids = get_pronoun_idx(pronoun_data['reference_sentence'], pronoun_list) #use this for sentence pronoun-ids if there is no context
			pos_sentence_pids = pos_context_pids[-len(pos_sentenceonly_pids):]	
			pos_pronoun_input = [pos_sentence_pids] 
		
			#common reference context
			neg_context = pronoun_data['reference_context'][:-ref_sen_len]
			neg_context.extend(pronoun_data['noisy_sentence'])
			neg_context_pids = get_pronoun_idx(neg_context, pronoun_list)
			neg_sentenceonly_pids = get_pronoun_idx(pronoun_data['noisy_sentence'], pronoun_list)
			neg_sentence_pids = neg_context_pids[-len(neg_sentenceonly_pids):]

			neg_context_input = [neg_context]		
			neg_sentence_input = [] 
			neg_pronoun_input = [neg_sentence_pids]
			

			try:
				# order of calling does not matter
				neg_score = net.forward(neg_context_input, neg_sentence_input, neg_pronoun_input, 1)
				pos_score = net.forward(pos_context_input, pos_sentence_input, pos_pronoun_input, 1)
			except:
				print(pronoun_data)
				raise
		
		
			if pos_score > neg_score:
				pos += 1
			
			elif pos_score == neg_score:
				
				tie += 1
			elif neg_score > pos_score:
	
				neg += 1
		except EOFError:
			break
	dataint.close()

	
	return pos, neg, tie, (pos+neg+tie)





def pairwise_loss(pos, neg, batch_size, margin=0.1, tie=False):
	if tie:
		return 0.0
	else:
		loss_zeros = torch.zeros(batch_size, 1, 1)
		loss_zeros = loss_zeros.to(device)
		loss = torch.max(margin + neg - pos, loss_zeros)
	
		return torch.mean(loss)


def padding_and_mask(input_vector, max_size):
	input_length = len(input_vector)
	padding_size = max_size - input_length
	padding = torch.zeros(padding_size, dtype=torch.long)
	mask = []
	mask.extend(torch.zeros(input_length))
	mask.extend(torch.ones(padding_size))

	input_vector = torch.cat((torch.LongTensor(input_vector), padding),0)

	assert len(input_vector) == max_size
	assert len(mask) == max_size

	return input_vector, mask, input_length


def get_pronoun_idx(input_sent, pronoun_list):
	return [i+1 for i in range(len(input_sent)) if input_sent[i] in pronoun_list.keys()]


dev_path = sys.argv[2]

#Adjust if needed
max_context_length = 304 
max_sentence_length = 135 
max_pronoun_length = 30 

torch.manual_seed(100)
torch.cuda.manual_seed_all(100)

dimensions = 1024 #based on ELMo
epochs = 20
org_batch_size = 30
batch_size = org_batch_size
learning_rate = 0.01
device = torch.device('cuda') #('cpu')

start = time.time()

best_epoch, best_accuracy =0,0.0


margin = 0.1

net = model.PronounScoreShared(dimensions, max_context_length, max_sentence_length, max_pronoun_length, margin)
net = net.to(device)
count = 0

pint = open("pronoun_idx", 'rb') #list of pronouns
pronoun_list = pickle.load(pint)


for epoch in range(epochs):
	count = 0
	if epoch >= 3:
		learning_rate = learning_rate/2
	
	optimizer=torch.optim.SGD(net.parameters(), lr=learning_rate)

	data_path = open(sys.argv[1], 'rb')
	EOFFLAG=False
	batch_size = org_batch_size
	running_loss = 0.0

	#Note that since the training data is large the pickle file is written to/read from one sample at a time
	while data_path:

		optimizer.zero_grad()

		pos_context_input = [] 
		pos_sentence_input = []
		pos_pronoun_input = []
		neg_context_input = [] 
		neg_sentence_input = []
		neg_pronoun_input = []

		try:
			for each_sample in range(batch_size):

				pronoun_data = pickle.load(data_path)
			
				ref_sen_len = len(pronoun_data['reference_sentence'])	
				pos_context = pronoun_data['reference_context']
				pos_sentence = pronoun_data['reference_sentence']
				pos_context_pids = get_pronoun_idx(pos_context, pronoun_list)
				pos_sentenceonly_pids = get_pronoun_idx(pos_sentence, pronoun_list) 
				pos_sentence_pids = pos_context_pids[-len(pos_sentenceonly_pids):]
				
				sys_sen_len = len(pronoun_data['system_sentence'])

				neg_context = pronoun_data['reference_context'][:-ref_sen_len]
				neg_context.extend(pronoun_data['system_sentence'])
	
				neg_sentence = pronoun_data['system_sentence']
				neg_context_pids = get_pronoun_idx(neg_context, pronoun_list)
				neg_sentenceonly_pids = get_pronoun_idx(neg_sentence, pronoun_list)
				neg_sentence_pids = neg_context_pids[-len(neg_sentenceonly_pids):]
				neg_pronouns = neg_sentence_pids 
		
				pos_context_input.append(pos_context) 
				pos_sentence_input.append(pos_sentence) 
				pos_pronoun_input.append(pos_sentence_pids)

				neg_context_input.append(neg_context) 
				neg_sentence_input.append(neg_sentence) 
				neg_pronoun_input.append(neg_sentence_pids)
			
				
		except EOFError:
			EOFFLAG=True
			break

		rand_idx = torch.randperm(batch_size)
		
		pos_context_batch = [pos_context_input[i] for i in rand_idx] 
		pos_sentence_batch = [] 
		pos_pronoun_batch = [pos_pronoun_input[i] for i in rand_idx]
	
		neg_context_batch = [neg_context_input[i] for i in rand_idx] 
		neg_sentence_batch = []
		neg_pronoun_batch = [neg_pronoun_input[i] for i in rand_idx] 

		pos_score = net.forward(pos_context_batch, pos_sentence_batch,  pos_pronoun_batch, batch_size)
		neg_score = net.forward(neg_context_batch, neg_sentence_batch, neg_pronoun_batch, batch_size)
		
		loss = pairwise_loss(pos_score, neg_score, batch_size)
		running_loss += loss.item()

		loss.backward()
		optimizer.step()
									   
		#Keeping track of progress
		#if count%100==0:
		#	print(count, end=".", flush=True)
		#count+=1

		if EOFFLAG:
			break

	pos, neg, tie, total = check_accuracy(net, dev_path)
	print(total)
	data_path.close()
	total_loss = running_loss/batch_size
	elapsed = time.time()-start
	recall = pos/total
	precision = pos/(pos+neg)
	f1 = 2 * precision * recall / (precision + recall)
	accuracy =  pos/total


	if accuracy >  best_accuracy:
		best_accuracy = accuracy
		best_epoch = epoch



	print("epoch=", epoch, "\t time=", elapsed, "\t dev_accuracy=", accuracy, "\tbest_acc:", best_accuracy, "\tbest_ep:", best_epoch)
	print()
	torch.save({'epoch':epochs, 'dev_acc':accuracy,'model_state_dict':net.state_dict(), 'optimizer_state_dict':optimizer.state_dict(), 'loss': running_loss, 'bsize':batch_size, 'dimensions':dimensions},"anaphora_model_"+str(epoch))

	










