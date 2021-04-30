import numpy as np
import torch.utils.data as data_utils

def word_to_char(id_to_word, char_to_id):
    max_len = 0
    for word in id_to_word.values():
        if max_len < len(word):
            max_len = len(word)
    max_len += 2        # because of 'sow' and 'eow'.
    word2char = []
    for idx in id_to_word:
        word = '{' + id_to_word[idx] + '}'      # '{' is a start-of-the-word
        left = (max_len-len(word)) // 2
        right = (max_len-len(word)) - left
        temp = [0]*left
        for c in word:
            temp.append(char_to_id[c])
        temp += [0]*right
        word2char.append(np.array(temp))
    return np.array(word2char)


def convert_to_char_level(corpus, word2char):
    fin = []
    for sentence in corpus:
        fin.append(word2char[sentence])
    return np.array(fin)


def make_seq(data, seq_len):
    seq_num = (len(data)-1) // seq_len      # except last word for source and first word for label.
    temp = []
    for i in range(seq_num):
        temp.append(data[i*seq_len: (i+1)*seq_len])
    return np.array(temp), seq_num


def make_batch(source, label, batch_size):
    tensor = data_utils.TensorDataset(source, label)
    loader = data_utils.DataLoader(tensor, batch_size=batch_size, shuffle=False)
    return loader
