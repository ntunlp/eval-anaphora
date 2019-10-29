import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import matplotlib.pyplot as plt
from torch.autograd import Variable
from allennlp.modules.elmo import Elmo, batch_to_ids

weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5'
big_weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
med_weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5'

options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json'
big_options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'
med_options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json'




class PronounScoreShared(nn.Module):
	def __init__(self, dimensions, max_context_length, max_sentence_length, max_pronoun_length, margin): 
		super(PronounScoreShared, self).__init__()

		self.dimensions = dimensions
		self.hidden_size = 512
		
		self.linear_layer_1 = nn.Linear(self.dimensions,1)
		self.linear_layer_2 = nn.Linear(max_pronoun_length,1)	
		self.elmo = Elmo(big_options_file, big_weight_file, 2, dropout=0, requires_grad=False) 	
		self.loss_margin = margin

		self.max_context_length = max_context_length
		self.max_sentence_length = max_sentence_length
		self.max_pronoun_length = max_pronoun_length
		
	#taken from the Transformer; not used
	def self_attention_layer(self, query, key, value, mask=None):
		scaled_score = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(self.dimensions)

		if mask is not None:
			scaled_score = scaled_score.masked_fill(mask, -np.inf)

		attention_weights = F.softmax(scaled_score, dim=-1)
		final_reps = torch.matmul(attention_weights, value)
		return attention_weights, final_reps
	
	def pronoun_query_attention_layer(self, query, key, value):
		scaled_score = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(self.dimensions)
		attention_weights = F.softmax(scaled_score, dim=-1)
		final_reps = torch.matmul(attention_weights, value)
		return attention_weights, final_reps

	#taken from the Transformer 
	def add_positional_encoding(self, rep, max_input_length):
		"Implement the PE function."
		

		# Compute the positional encodings once in log space.
		pe = torch.zeros(max_input_length, self.dimensions)
		position = torch.arange(0., max_input_length).unsqueeze(1)
		div_term = torch.exp(torch.arange(0., self.dimensions, 2) * -(math.log(10000.0) / self.dimensions))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)
		
		temp = Variable(self.pe[:, :rep.size(1)], requires_grad=False)
		temp = temp.cuda()
		rep = rep + temp

		return rep
	
	
	def layer_normalization(self, input_vector, eps=1e-6):
	
		self.a_2 = torch.ones(self.dimensions)
		self.a_2 = self.a_2.cuda()
	
		self.b_2 = torch.zeros(self.dimensions)
		self.b_2 = self.b_2.cuda()
		self.eps = eps
		
		input_vector = input_vector.cuda()
		mean = input_vector.mean(-1, keepdim=True)
		mean = mean.cuda()
		std = input_vector.std(-1, keepdim=True)
		std = std.cuda()

		return self.a_2 * (input_vector - mean) / (std + self.eps) + self.b_2

	def forward(self, input_context, input_sentence, input_pronouns,  bs, mask=None, tie=False):
		
		querybatch2ids = batch_to_ids(input_context) #change to input sentence for no-context version
		querybatch2ids = querybatch2ids.cuda()
		query = self.elmo(querybatch2ids)['elmo_representations'][0]

		batch_prn_states = []
		for x in range(bs):
			prn_states = torch.stack([query[x][i-1] for i in input_pronouns[x] if i!=0])
			plen = prn_states.size(0)
			padding = torch.zeros(self.max_pronoun_length - plen, self.dimensions)
			padding = padding.cuda()
			padded_prn_states = torch.cat((prn_states, padding),0)
			batch_prn_states.append(padded_prn_states)
		batch_tensor_prn_state = torch.Tensor()
		batch_tensor_prn_state = torch.stack(batch_prn_states)
	
		query = self.add_positional_encoding(query, self.max_context_length)
			
		self.pquery_attn_wts, pquery_rep = self.pronoun_query_attention_layer(batch_tensor_prn_state, query, query)

		#residual connection
		pquery_rep = self.layer_normalization(batch_tensor_prn_state+pquery_rep)
		
		pronoun_scores = self.linear_layer_1(pquery_rep)
	
		score = self.linear_layer_2(pronoun_scores.view(-1,self.max_pronoun_length))
	
		return score

