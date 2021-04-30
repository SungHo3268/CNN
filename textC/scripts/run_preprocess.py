import argparse
from distutils.util import strtobool as _bool
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import json
import time
import numpy as np
import torch
import pickle
import time
import os
import sys
sys.path.append(os.getcwd())
from src.preprocess import *


########## make pre-trained weight and dictionary from word2vec ##########
# save_word2vec()


################# load pre-trained weight and dictionary #################
pre_W = load_preW()
word_to_id, id_to_word = load_dictionary()


############################## load dataset ##############################
load_dir = 'datasets/raw/'
save_dir = 'datasets/preprocessed/'
datasets = {0:'MR',
            1:'SST-1',
            2:'SST-2',
            3:'Subj',
            4:'TREC',
            5:'CR',
            6:'MPQA'}

for i in range(len(datasets)):
    print('\n', datasets[i])
    data = data_load(load_dir, datasets[i])
    print('#sentences: ', len(data[0])+len(data[1])+len(data[2]))
    text, label = split_text_label(data)
    tokens = tokenize(text ,datasets[i])
    W, word2id, id2word = shrink_weight(pre_W, word_to_id, id_to_word, tokens)
    save_weightNdict(save_dir, datasets[i], W, word2id, id2word)
    temp_corpus = make_corpus(tokens, word2id)
    text = padding(temp_corpus, pad=0, middle=True)
    corpus = (text, label)
    save_corpus(corpus, save_dir, datasets[i])

'''
pre_weight = W
wordDict = (word2id, id2word) 
corpus = (train, dev, test) = 0.81:0.09:0.1, zero_padding
'''

