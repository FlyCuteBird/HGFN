'''Bert Embeddings, return: sum_last_4_layers'''

import torch
import numpy as np
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

class Bert_Embeddings(nn.Module):
    def __init__(self):
        super(Bert_Embeddings, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def output_embedding(self, x):

        batchSizeEmbed  = []
        for i in range(len(x)):
            tokens_tensor = x[i].unsqueeze(0)
            #tokens_tensor = torch.tensor([x[i]])
            with torch.no_grad():
                last_hidden_states = self.model(tokens_tensor)[0]
            token_embed = []
            for token_i in range(len(x[i])):
                hidden_layers = []
                for layer_i in range(len(last_hidden_states)):
                    vec = last_hidden_states[layer_i][0][token_i]
                    hidden_layers.append(vec)
                token_embed.append(hidden_layers)
            sum_last_4_layers = [torch.sum(torch.stack(layer)[-4:], 0) for layer in token_embed]
            sum_last_4_layers = [i.cuda().data.cpu().numpy().tolist() for i in sum_last_4_layers]
            #sum_last_4_layers = [i.numpy().tolist() for i in sum_last_4_layers]
            batchSizeEmbed.append(sum_last_4_layers)
            token_embed = []
            sum_last_4_layers = []
        return torch.Tensor(batchSizeEmbed).cuda()
    def forward(self, x):
        x = self.output_embedding(x)
        return x
