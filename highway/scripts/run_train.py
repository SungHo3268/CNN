import argparse
from distutils.util import strtobool as _bool
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data_utils
import pickle
import time
import sys
import os
sys.path.append(os.getcwd())
from src.functions import *
from src.layers import *
from src.model import *


############################ Argparse ############################
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--dataset', type=str, default='ptb', help='ptb | cs | de | es | fr | ru')
parser.add_argument('--lr', type=float, default=1.0)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--seq_len', type=int, default=35)
parser.add_argument('--char_dim', type=int, default=15)
parser.add_argument('--highway_layer', type=int, default=1)
parser.add_argument('--lstm_dim', type=int, default=300)
parser.add_argument('--lstm_layer', type=int, default=2)
parser.add_argument('--max_epoch', type=int, default=25)        # or 35
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--max_norm', type=float, default=5.0)
parser.add_argument('--gpu', type=_bool, default=False)
args = parser.parse_args()
log_dir = 'log/{}'.format(args.dataset)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
with open(log_dir + 'argparse.json', 'w') as f:
    json.dump(args.__dict__, f)


############################ Load Data ############################
load_dir = 'datasets/preprocessed/{}'.format(args.dataset)
with open(os.path.join(load_dir, 'vocab.pkl'), 'rb') as fr:
    char_to_id, id_to_char, word_to_id, id_to_word = pickle.load(fr)

with open(os.path.join(load_dir, 'corpus.pkl'), 'rb') as fr:
    train, valid, test = pickle.load(fr)            # train = (num_tokens, )


############################ Hyperparameter ############################
V = len(word_to_id)
C = len(char_to_id)
T = len(train)
kernel_w = [1, 2, 3, 4, 5, 6]
h = np.multiply(kernel_w, 25)


############################ Input and label ############################
# split with sequence length
train, seq_num = make_seq(train, args.seq_len)
valid, _ = make_seq(valid, args.seq_len)
test, _ = make_seq(test, args.seq_len)

# convert word id to char id
word2char = word_to_char(id_to_word, char_to_id)
train_char = torch.from_numpy(word2char[train])     # train_char = (seq_num, seq_len, max_word_length)
valid_char = torch.from_numpy(word2char[valid])
test_char = torch.from_numpy(word2char[test])

# make input source and label
train_input = train_char[:-1]                       # train_input = (seq_num-1, seq_len, max_word_length)
valid_input = valid_char[:-1]
test_input = test_char[:-1]
train_label = torch.from_numpy(train[1:])           # train_input = (seq_num-1, seq_len)
valid_label = torch.from_numpy(valid[1:])
test_label = torch.from_numpy(test[1:])

# make batch
train_loader = make_batch(train_input, train_label, args.batch_size)
valid_loader = make_batch(valid_input, valid_label, args.batch_size)
test_loader = make_batch(test_input, test_label, args.batch_size)


############################ Init Net ############################
model = CharLM(V, C, args.char_dim, kernel_w, h, args.lstm_dim, args.lstm_layer, args.dropout)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=args.lr)
# optimizer = optim.Adam(model.parameters(), lr=0.001)

h_0 = torch.zeros(args.lstm_layer, args.batch_size, args.lstm_dim)
c_0 = torch.zeros(args.lstm_layer, args.batch_size, args.lstm_dim)
hidden = (torch.autograd.Variable(h_0), torch.autograd.Variable(c_0))

device = None
if args.gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)


############################ train ############################
total_loss = 0
count = 0
loss_list = []
prev_ppl = V
best_ppl = len(valid_input)
val_loss_list = []
val_ppl_list = []
for epoch in range(args.max_epoch):
    model.train()
    for source, label in tqdm(train_loader,
                              desc='epoch: {}/{}'.format(epoch+1,args.max_epoch), bar_format="{l_bar}{bar:20}{r_bar}"):
        if source.shape[0] != args.batch_size:
            continue
        if args.gpu:
            source = source.to(device)
            label = label.to(device)
            hidden = [state.to(device) for state in hidden]

        hidden = [state.detach() for state in hidden]
        optimizer.zero_grad()
        out, hidden = model(source, hidden)
        loss = criterion(out.view(-1, V), label.view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm, norm_type=2)
        optimizer.step()

        total_loss += loss.data
        count += 1
    avg_loss = total_loss/count
    loss_list.append(avg_loss)
    total_loss=0
    count = 0

    # validation per epoch
    model.eval()
    batch_loss = []
    batch_ppl = []
    valid_hidden = hidden
    for source, label in tqdm(valid_loader,
                    desc='validation: ', bar_format="{l_bar}{bar:20}{r_bar}"):
        if source.shape[0] != args.batch_size:
            continue
        if args.gpu:
            source = source.to(device)
            label = label.to(device)
            valid_hidden = [state.to(device) for state in valid_hidden]
        valid_hidden = [state.detach() for state in valid_hidden]
        out, valid_hidden = model(source, valid_hidden)
        loss = criterion(out.view(-1, V), label.view(-1))
        batch_loss.append(float(loss))
    valid_loss = np.mean(batch_loss)
    valid_ppl = np.exp(valid_loss)

    val_loss_list.append(valid_loss)
    val_ppl_list.append(valid_ppl)
    print("valid loss: {}  |  PPL: {}\n".format(valid_loss, valid_ppl))

    if best_ppl > valid_ppl:
        best_ppl = valid_ppl
    if (prev_ppl - valid_ppl) <= 1:
        args.lr /= 2
        print("The learning rate is halved")
    prev_ppl = valid_ppl
print("best ppl for validation: ", best_ppl)


############################ plot valid ############################
# plot train and validation graph
fig, ax1 = plt.subplots()
ax1.plot(val_loss_list, color='red')
ax2 = ax1.twinx()
ax2.plot(val_ppl_list, color='green')

ax1.set_xlabel('epochs')
ax1.set_ylabel('val_loss')
ax2.set_ylabel('val_acc')
plt.show()


############################ test ############################
model.eval()
test_loss_list = []
test_hidden = hidden
for source, label in tqdm(test_loader,
                desc='test: ', bar_format="{l_bar}{bar:20}{r_bar}"):
    if source.shape[0] != args.batch_size:
        continue
    if args.gpu:
        source = source.to(device)
        label = label.to(device)
        test_hidden = [state.to(device) for state in test_hidden]
    test_hidden = [state.detach() for state in test_hidden]
    out, test_hidden = model(source, test_hidden)
    loss = criterion(out.view(-1, V), label.view(-1))
    test_loss_list.append(float(loss))
test_loss = np.mean(test_loss_list)
test_ppl = np.exp(test_loss)
print("test loss: {}  |  test PPL: {}\n".format(test_loss, test_ppl))

