import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils


class Embedding(nn.Module):
    def __init__(self, V, D, pre_weight, model_type):
        """
        :param V: vocabulary size
        :param D: embedding dimension (use 100dim pre_trained billion word weight)
        :param pre_weight: pre-trained weight
        :param model_type: {'rand' | 'static' | 'non-static' | 'multichannel'}
        """ 
        super(Embedding, self).__init__()
        self.V = V
        self.D = D
        self.embedding = nn.ModuleList()        # There are many types of Embedding.
        if model_type.lower() == 'rand':
            weight = torch.distributions.Uniform(low=-0.25, high=0.25).sample((self.V, self.D))
            embedding_layer = nn.Embedding.from_pretrained(weight, freeze=False)
            # embedding_layer = nn.Embedding(num_embeddings=self.V, embedding_dim=self.D, padding_idx=0)
            self.embedding.append(embedding_layer)
        elif model_type.lower() == 'static':
            embedding_layer = nn.Embedding.from_pretrained(pre_weight, freeze=True)      # already padding done. in pre_weight
            self.embedding.append(embedding_layer)
        elif model_type.lower() == 'non-static':
            embedding_layer = nn.Embedding.from_pretrained(pre_weight, freeze=False)
            self.embedding.append(embedding_layer)
        elif model_type.lower() == 'multichannel':
            embedding_layer1 = nn.Embedding.from_pretrained(pre_weight, freeze=True)
            embedding_layer2 = nn.Embedding.from_pretrained(pre_weight, freeze=False)
            self.embedding.append(embedding_layer1)
            self.embedding.append(embedding_layer2)
        else:
            print('Wrong model_type!!!!, please')
            # exit()
