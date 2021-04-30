import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from tqdm.auto import tqdm


class CharCNN(nn.Module):
    def __init__(self, C, D, kernel_w, h):
        """
        :param C: character vocabulary size
        :param D: char embedding dimension = 15
        :param kernel_w: kernel width = [1, 2, 3, 4, 5, 6]
        :param h: the number of kernel = [25, 50, 75, 100, 125, 150]
        """
        super(CharCNN, self).__init__()
        self.C = C
        self.D = D
        self.kernel_W = kernel_w
        self.h = h

        self.embedding = nn.Embedding(self.C, self.D, padding_idx=0)
        self.embedding.weight.data.uniform_(-0.05, 0.05)

        self.conv = nn.ModuleList()
        for hh, k in zip(h, kernel_w):
            conv = nn.Conv2d(1, hh, (k, self.D), bias=True)
            conv.weight.data.uniform_(-0.05, 0.05)
            conv.bias.data.fill_(0)
            self.conv.append(conv)


    def forward(self, x):
        """
        :param x = (batch_size, sequence_length, max_word_length)
        """
        batch_size, sequence_length, max_word_length = x.shape
        x = x.view(-1, max_word_length)      # (#total words, max_word_length)
        x = self.embedding(x)  # x = (#total words, max_word_length, embedding_dim)
        x = x.unsqueeze(1)  # x = (#total words, 1, max_word_length, embedding_dim) = (-1, 1, 21, 15)
        out = [torch.tanh(conv(x)).squeeze() for conv in self.conv]  # out = [(#total words, filter_num, seq_size-kernel_width+1), ]
        out = [F.max_pool1d(i, i.size(-1)).squeeze() for i in out]  # out = [(#total words,, filter_num), ]
        out = torch.cat(out, dim=1)      # out = (#total words,, sum(filter_size)) = (-1, 525)
        out = out.view(batch_size, sequence_length, -1)
        return out


class Highway(nn.Module):
    def __init__(self, linear_size):
        """
        :param linear_size: the size of square weight
        """
        super(Highway, self).__init__()
        self.linear1 = nn.Linear(linear_size, linear_size, bias=True)
        self.linear2 = nn.Linear(linear_size, linear_size, bias=True)
        self.linear1.weight.data.uniform_(-0.05, 0.05)
        self.linear2.weight.data.uniform_(-0.05, 0.05)
        self.linear1.bias.data.fill_(0)
        self.linear2.bias.data.fill_(0)

    def forward(self, x):
        t = torch.sigmoid(self.linear1(x))
        z = torch.mul(t, F.relu(self.linear2(x))) + torch.mul(1-t, x)
        return z

