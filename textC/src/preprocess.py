import numpy as np
import gensim
import pickle
from tqdm import tqdm
import os
import collections
import re
from collections import Counter
import os
import sys
sys.path.append(os.getcwd())


def save_word2vec():
    path = "datasets/preprocessed/GoogleNews-vectors-negative300.bin"
    model = gensim.models.KeyedVectors.load_word2vec_format(path, binary = True)

    word_to_id = {}
    id_to_word = {}
    W = model.vectors.astype("float32")
    for word in model.vocab:
        word_to_id[word] = len(word_to_id)
        id_to_word[len(id_to_word)] = word

    with open("datasets/preprocessed/wordDict_all.pkl", "wb") as fw:
        pickle.dump((word_to_id, id_to_word), fw)

    with open("datasets/preprocessed/pre_weight_all.pkl", "wb") as fw:
        pickle.dump(W, fw)
        
        
def load_dictionary():
    with open('datasets/preprocessed/wordDict_all.pkl', 'rb') as fr:
        word_to_id, id_to_word = pickle.load(fr)
    return word_to_id, id_to_word


def load_preW():
    with open('datasets/preprocessed/pre_weight_all.pkl', 'rb') as fr:
        pre_W = pickle.load(fr)
    return pre_W


def data_load(dataset_dir, dataset):
    path = os.path.join(dataset_dir, dataset)
    with open(os.path.join(path, dataset+'_train'), 'r') as f:
        train = f.readlines()
    with open(os.path.join(path, dataset+'_dev'), 'r') as f:
        dev = f.readlines()
    with open(os.path.join(path, dataset+'_test'), 'r') as f:
        test = f.readlines()
    return train, dev, test


def split_text_label(data):
    texts = []
    labels = []
    for d in data:
        text = []
        label = []
        for line in d:
            label.append(int(line[:1]))
            text.append(line[2:])
        texts.append(text)
        labels.append(label)
    return texts, labels


def tokenizer(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower().split()


def SST_tokenizer(string):
  string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
  string = re.sub(r"\s{2,}", " ", string)
  return string.strip().lower().split()
  

def tokenize(data, dataset):
    if dataset[:3] == 'SST':
        train, dev, test = data
        for i, line in enumerate(train):
            train[i] = SST_tokenizer(line)
        for i, line in enumerate(dev):
            dev[i] = SST_tokenizer(line)
        for i, line in enumerate(test):
            test[i] = SST_tokenizer(line)
        data = (train, dev, test)
        return data
    else:
        train, dev, test = data
        for i, line in enumerate(train):
            train[i] = tokenizer(line)
        for i, line in enumerate(dev):
            dev[i] = tokenizer(line)
        for i, line in enumerate(test):
            test[i] = tokenizer(line)
        data = (train, dev, test)
        return data


def shrink_weight(pre_W, word_to_id, id_to_word, tokens):
    word2id = dict()
    id2word = dict()
    word2id['<pad>'] = 0
    id2word[0] = '<pad>'
    in_vocab = set()
    no_vocab = set()
    for data in tokens:
        for line in data:
            for word in line:
                if word in word_to_id:
                    in_vocab.add(word_to_id[word])
                else:
                    no_vocab.add(word)
    in_vocab = list(in_vocab)
    no_vocab = list(no_vocab)
    print('in_vocab: ', len(in_vocab),'\nno_vocab: ', len(no_vocab))
    W = np.zeros((1, pre_W.shape[1]))
    W = np.append(W, pre_W[in_vocab], axis=0)
    # W = np.append(W, np.random.uniform(low=-0.01, high=0.01, size=(len(no_vocab), pre_W.shape[1])), axis=0)
    W = np.append(W, np.random.uniform(low=-0.25, high=0.25, size=(len(no_vocab), pre_W.shape[1])), axis=0)

    for idx in in_vocab:
        word2id[id_to_word[idx]] = len(word2id)
        id2word[len(id2word)] = id_to_word[idx]
    for word in no_vocab:
        word2id[word] = len(word2id)
        id2word[len(id2word)] = word
    return W, word2id, id2word


def save_weightNdict(dataset_dir, dataset, W, word2id, id2word):
    path = os.path.join(dataset_dir, dataset)
    with open(os.path.join(path, 'pre_weight.pkl'), 'wb') as fw:
        pickle.dump(W, fw)
    with open(os.path.join(path, 'wordDict.pkl'), 'wb') as fw:
        pickle.dump((word2id, id2word), fw)


def make_corpus(tokens, word2id):
    for data in tokens:
        for i, line in enumerate(data):
            temp = []
            for word in line:
                temp.append(word2id[word])
            data[i] = temp
    return tokens


def padding(corpus, pad=0, middle=True):
    for data in tqdm(corpus, desc='zero padding', bar_format="{l_bar}{bar:20}{r_bar}"):
        max_len = 0
        for line in data:
            if len(line) > max_len:
                max_len = len(line)
        for i, line in enumerate(data):
            dif = max_len - len(line)
            if middle:
                left = dif // 2
                right = dif - left
                data[i] = [pad]*left + line + [pad]*right
            else: data[i] = line + [pad]*dif
    return corpus


def save_corpus(corpus, dataset_dir, dataset):
    path = os.path.join(dataset_dir, dataset)
    with open(os.path.join(path, 'corpus.pkl'), 'wb') as fw:
        pickle.dump(corpus, fw)

