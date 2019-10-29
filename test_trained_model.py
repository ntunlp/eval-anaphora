import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import matplotlib.pyplot as plt
import pickle
import sqlite3
import re
import sys
import random
from tqdm import tqdm
from torch.autograd import Variable
from collections import OrderedDict

import model

def get_pronoun_idx(input_sent, pronoun_list):
	return [i+1 for i in range(len(input_sent)) if input_sent[i] in pronoun_list.keys()]


def get_pronoun_score(common_context, input_sentence, net):
	input_context = common_context + " " + input_sentence

	input_tokens = input_context.split()
	input_context_batch = [input_tokens]
	input_context_pids = get_pronoun_idx(input_tokens, pronoun_list)

	input_sentence_tokens = input_sentence.split()
	input_sentence_batch = [input_sentence_tokens]
	input_sentenceonly_pids = get_pronoun_idx(input_sentence_tokens, pronoun_list)

	input_pids = input_context_pids[-len(input_sentenceonly_pids):]
	
	input_pids_batch = [input_pids]

	score = net.forward(input_context_batch, input_sentence_batch, input_pids_batch, 1)
	return score

#Adjust if needed
max_context_length = 304
max_sentence_length = 135 
max_pronoun_length = 30

torch.manual_seed(100)
torch.cuda.manual_seed_all(100)
random.seed(100)

dimensions = 1024

device = torch.device('cpu') #('cuda')

test_data = open(sys.argv[2], 'rb')
test_samples = pickle.load(test_data)

margin = 0.1

net = model.PronounScoreShared(dimensions, max_context_length, max_sentence_length, max_pronoun_length, margin) 

model_path = sys.argv[1]
start = time.time()

model_dict = torch.load(model_path, map_location='cpu') #remove map location if running on GPU
print(model_dict.keys())
state_dict = torch.load(model_path, map_location='cpu')['model_state_dict']

# create new OrderedDict that does not contain `module.`
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if k!='pe': # remove `module.`
        new_state_dict[k] = v

net.load_state_dict(new_state_dict)
net.eval()
net = net.to(device)

pint = open("pronoun_idx", 'rb')
pronoun_list = pickle.load(pint)

outint = open("trained_model_output_"+sys.argv[2], 'wb')

#Note that this code expects a pickle file with the common reference context provided separately. Since the model is trained pairwise against the reference, compare two system translation
#candidates by running both through the net, and subtracting the scores from the reference score. The one with the lower score is closer to the reference (and therefore considered
# as higher ranked)

for n, each_sample in tqdm(enumerate(test_samples)):
	
	common_reference_context = each_sample['common_reference_context']
	reference_sentence = each_sample['reference_sentence']
	candidate_sentence = each_sample['candidate_sentence']

	temp_dict = {}
	temp_dict['common_ref_context'] = common_reference_context
	temp_dict['sample_id'] = n
	temp_dict['reference_sentence'] = reference_sentence
	temp_dict['candidate_sentence'] = candidate_sentence

	ref_score = get_pronoun_score(common_reference_context, reference_sentence, net)
	cand_score = get_pronoun_score(common_reference_context, candidate_sentence, net)

	temp_dict['reference_score'] = ref_score
	temp_dict['candidate_score'] = cand_score

	pickle.dump(temp, outint)

	


	
	



	



