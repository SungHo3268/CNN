import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from tqdm.auto import tqdm
import sys
import os

sys.path.append(os.getcwd())
from src.layers import *


class CharLM(nn.Module):
    def __init__(self, V, C, D, kernel_w, h, lstm_dim, lstm_layer, dropout):
        """
        :param V: word vocabulary size
        :param C: character vocabulary size
        :param D: char embedding dimension = 15
        :param kernel_w: kernel width = [1, 2, 3, 4, 5, 6]
        :param h: the number of kernel = [25, 50, 75, 100, 125, 150]
        :param max_norm: the max norm of l2 regularization
        :param lstm_dim: the hidden dimension of lstm layer
        :param lstm_layer: the number of layer of lstm
        :param dropout: the ratio of dropout layer
        """
        super(CharLM, self).__init__()
        linear_size = sum([filter_num for filter_num in h])

        self.charCNN = CharCNN(C, D, kernel_w, h)
        self.highway = Highway(linear_size)
        self.lstm = nn.LSTM(linear_size, lstm_dim, num_layers=lstm_layer, bias=True, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(lstm_dim, V)
        self.fc.weight.data.uniform_(-0.05, 0.05)
        self.fc.bias.data.fill_(0)

    def forward(self, x, hidden):
        """
        :param x: (batch_size, seq_len, char_dim)
        :param hidden: (h_0, c_0)
        """
        batch_size, seq_len, _ = x.shape
        x = self.charCNN(x)  # x = (batch_size, seq_len, total_num_filters)
        x = self.highway(x)  # x = (batch_size, seq_len, total_num_filters)
        x, h = self.lstm(x, hidden)  # x = (batch_size, seq_len, lstm_dim)   /   hidden = (h, c)
        out = self.dropout(x)  # out = (batch_size, seq_len, lstm_dim)
        out = self.fc(out)  # out = (batch_size, seq_len, V)
        return out, h

