import argparse
from distutils.util import strtobool as _bool
import json
import torch
import torch.utils.data as data_utils
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pickle
import time
import sys
import os
sys.path.append(os.getcwd())
from src.models import *


################################ Argparse ################################
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='dummy')
parser.add_argument('--model_type', type=str, default='multichannel', help="rand | static | non-static | multichannel")
parser.add_argument('--dataset', type=str, default='TREC', help="MR | SST-1 | SST-2 | Subj | TREC | CR | MPQA")
parser.add_argument('--optimizer', type=str, default='Adadelta', help='Adadelta | Adam')
parser.add_argument('--emb_dim', type=int, default=300)
parser.add_argument('--max_epoch', type=int, default=25)
parser.add_argument('--batch_size',type=int, default=50)
parser.add_argument('--drop_out', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--adam_decay', type=float, default=1e-03)
parser.add_argument('--adadelta_decay', type=float, default=0.95)
parser.add_argument('--cuda', type=_bool, default=False)
 
args = parser.parse_args()
log_dir = 'log/{}_{}'.format(args.dataset, args.model_type)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
with open(log_dir + 'argparse.json', 'w') as f:
    json.dump(args.__dict__, f)


############################## load preprocessed data #################################
dataset_dir = 'datasets/preprocessed/{}/'.format(args.dataset)

# load dictionary
with open(os.path.join(dataset_dir, 'wordDict.pkl'), 'rb') as fr:
    word2id, id2word = pickle.load(fr)

# load pre-trained word vectors
with open(os.path.join(dataset_dir, 'pre_weight.pkl'), 'rb') as fr:
    pre_weight = pickle.load(fr)
    pre_weight = torch.from_numpy(pre_weight).float()
# load corpus
with open(os.path.join(dataset_dir, 'corpus.pkl'), 'rb') as fr:
    text, label = pickle.load(fr)
train_text, dev_text, test_text = text
train_label, dev_label, test_label = label


############################## Hyperparameter #################################
V = len(word2id)
channel_size = 1
if args.model_type == 'multichannel':
    channel_size = 2

class_num = len(set(train_label))       # output
kernel_size = [3, 4, 5]
feature_map_size = 100


############################## Input data #################################
train_text = torch.from_numpy(np.array(train_text))
dev_text = torch.from_numpy(np.array(dev_text))
test_text = torch.from_numpy(np.array(test_text))
train_label = torch.from_numpy(np.array(train_label))
dev_label = torch.from_numpy(np.array(dev_label))
test_label = torch.from_numpy(np.array(test_label))

train = data_utils.TensorDataset(train_text, train_label)

train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True)


############################## Init Net #################################
model = CNN_classifier(V, args.emb_dim, pre_weight, args.model_type,
                       channel_size, kernel_size, feature_map_size, class_num, args.drop_out)
optimizer = None
if args.optimizer == 'Adadelta':
    optimizer = optim.Adadelta(model.parameters(), rho=args.adadelta_decay, eps=1e-06)
elif args.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.adam_decay)
    # optimizer = optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss()

# if gpu
device = None
if args.cuda:
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    model.to(device)


############################## Train start #################################
count = 0
total_loss = 0
loss_list = []
val_acc_list = []
print("======================= {}_{}_{} ======================="
      .format(args.model_type, args.dataset, args.optimizer))
for epoch in range(args.max_epoch):
    model.train()
    for text, label in tqdm(train_loader, desc='{}/{}'.format(epoch+1, args.max_epoch), bar_format="{l_bar}{bar:20}{r_bar}"):
        if args.cuda:
            text.to(device)
            label.to(device)

        optimizer.zero_grad()
        pred = model(text)
        loss = criterion(pred, label)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3, norm_type=2.0)
        optimizer.step()

        total_loss += loss.data
        count += 1

    avg_loss = total_loss / count
    loss_list.append(avg_loss)
    total_loss = 0
    count = 0

    # validation
    model.eval()
    val_pred = torch.max(model(dev_text), 1)[1]
    correct = (val_pred.data == dev_label.data).sum()
    val_acc = correct/len(dev_label) * 100
    val_acc_list.append(val_acc)

    # print("epoch: {}/{}  |  loss: {}  |  val_score: {}".format(epoch+1, args.max_epoch, avg_loss, val_acc))
'''
################## plot ###################
# plot train and validation graph
fig, ax1 = plt.subplots()
ax1.plot(loss_list, color='red')
ax2 = ax1.twinx()
ax2.plot(val_acc_list, color='green')

ax1.set_xlabel('epochs')
ax1.set_ylabel('train_loss')
ax2.set_ylabel('val_acc')
plt.show()
##########################################
'''

# test
model.eval()
test_pred = torch.max(model(test_text), 1)[1]
correct = (test_pred.data == test_label.data).sum()
test_acc = correct/len(test_label) * 100

print("test_acc: {:0.1f}".format(test_acc))
print("max_val_acc: {:0.1f}".format(max(val_acc_list)))
print("========================================================\n\n".format(args.model_type, args.dataset))

