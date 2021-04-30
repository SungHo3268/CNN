import torch
import torch.nn as nn
import torch.functional as F
import pickle
import numpy as np
from tqdm.auto import tqdm
from collections import Counter
import sys
import os
sys.path.append(os.getcwd())


def data_load(log_dir, file):
    path = os.path.join(log_dir, file)
    with open(os.path.join(path, 'train.txt'), 'r') as f:
        train = f.readlines()
    with open(os.path.join(path, 'valid.txt'), 'r') as f:
        valid = f.readlines()
    with open(os.path.join(path, 'test.txt'), 'r') as f:
        test = f.readlines()

    if file =='ptb':
        for i, line in enumerate(train):
            train[i] = line + ' +'
        for i, line in enumerate(valid):
            valid[i] = line + ' +'
        for i, line in enumerate(test):
            test[i] = line + ' +'
    return train, valid, test


def make_vocab(data):
    char_vocab = set()
    word_to_id = {'<unk>':0}
    id_to_word = {0:'<unk>'}
    for line in tqdm(data, desc='Making vocab', bar_format='{l_bar}{bar:20}{r_bar}'):
        #make word dictionary
        for word in line.split():
            if word not in word_to_id:
                word_to_id[word] = len(word_to_id)
                id_to_word[len(id_to_word)] = word
            for c in word:
                char_vocab.add(c)
    # make char dictionary
    char_to_id = {'<pad>':0, '{':1, '}':2}
    id_to_char = {0:'<pad>', 1:'{', 2:'}'}
    for char in char_vocab:
        char_to_id[char] = len(char_to_id)
        id_to_char[len(id_to_char)] = char
    return char_to_id, id_to_char, word_to_id, id_to_word


def make_corpus(data, word_to_id):
    corpus = []
    for line in data:
        for word in line.split():
            corpus.append(word_to_id[word])
    return np.array(corpus)


def save_vocab(save_dir, file, char_to_id, id_to_char, word_to_id, id_to_word):
    path = os.path.join(save_dir, file)
    if not os.path.exists(path):
        os.mkdir(path)
    with open(os.path.join(path, 'vocab.pkl'), 'wb') as fw:
        pickle.dump((char_to_id, id_to_char, word_to_id, id_to_word), fw)
    print("Successfully save the vocabulary and dictionary.")


def save_corpus(save_dir, file, train, valid, test):
    path = os.path.join(save_dir, file)
    if not os.path.exists(path):
        os.mkdir(path)
    with open(os.path.join(path, 'corpus.pkl'), 'wb') as fw:
        pickle.dump((train, valid, test), fw)
    print("Successfully save the corpus.")
