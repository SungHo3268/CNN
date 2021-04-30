import argparse
from distutils.util import strtobool as _bool
import json
import numpy as np
from tqdm.auto import tqdm
import pickle
import sys
import os
sys.path.append(os.getcwd())
from src.preprocess import *


log_dir = 'datasets/raw'
save_dir = 'datasets/preprocessed'
data_list = {0: 'ptb',
             1: 'cs',
             2: 'de',
             3: 'es',
             4: 'fr',
             5: 'ru'}

for i in range(len(data_list)):
    file = data_list[i]
    print("=========", file, "=========")
    train, valid, test = data_load(log_dir, file)
    char_to_id, id_to_char, word_to_id, id_to_word = make_vocab(train)
    
    train_corpus = make_corpus(train, word_to_id)
    valid_corpus = make_corpus(valid, word_to_id)
    test_corpus = make_corpus(test, word_to_id)

    save_vocab(save_dir, file, char_to_id, id_to_char, word_to_id, id_to_word)
    save_corpus(save_dir, file, train_corpus, valid_corpus, test_corpus)
    print('\n')
