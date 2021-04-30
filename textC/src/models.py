import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.getcwd())
from src.layers import Embedding


class CNN_classifier(Embedding):
    def __init__(self, V, D, pre_weight, model_type,
                 channel_size, kernel_size, feature_map_size,
                 out, drop_out):
        """
        :param V: vocabulary size
        :param D: embedding dimension
        :param pre_weight: pre-trained weight; it already done middle zero padding.
        :param model_type: {'rand' | 'static' | 'non-static' | 'multichannel'}
        :param channel_size: the number of channel
        :param kernel_size: the list of kernel size, like [(3, 4, 5]
        :param feature_map_size: the number of feature maps for each kernel size (of 100)
        :param out: the number of class
        :param drop_out: dropout ratio (of 0.5)
        """

        super(CNN_classifier, self).__init__(V, D, pre_weight, model_type)
        ''' self.embedding = nn.ModuleList(embedding_layer) '''
        self.model_type = model_type
        self.conv = nn.ModuleList([nn.Conv2d(channel_size, feature_map_size, (kernel, D)) for kernel in kernel_size])
        for i, conv in enumerate(self.conv):
            conv.weight = torch.nn.Parameter(torch.distributions.Uniform(-0.01, 0.01).sample(
                (feature_map_size, channel_size, kernel_size[i], D)))
        self.dropout = nn.Dropout(drop_out)
        self.fc = nn.Linear(len(kernel_size) * feature_map_size, out)

    def forward(self, x):
        """
        x = (batch_size, sentence_max_length)
        """
        x = [emb(x) for emb in self.embedding]      # x= channel x [(batch_size, S, D)]
        x = torch.cat(x, 1)                         # x = (batch_size, S*channel, D)
        x = x.unsqueeze(1)                          # x = (batch_size, 1, S*channel, D)
        if self.model_type == 'multichannel':
            x = torch.tile(x, (1, 2, 1, 1))
        out = [F.relu(conv(x)).squeeze(3) for conv in self.conv]        # out = #kernel x (batch_size, 100, S-kernel_size+1)
        out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in out]      # out = #kernel x (batch_size, 100)
        out = torch.cat(out, 1)         # out = (batch_size, 100*#kernel)
        out = self.dropout(out)
        out = self.fc(out)              # out = (batch_size, #class)
        return out

    # def validation(self, ):