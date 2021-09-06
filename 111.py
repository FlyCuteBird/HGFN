import json
import collections
import torch
import numpy as np
import torch.nn as nn
embed = {}
lengths = 0
with open("glove300.txt", 'rb+') as f:
    for line in f.readlines():
        line_list = line.strip().split()
        word = line_list[0].decode('utf-8')
        embedding = line_list[1:]
        word_embed = [float(num) for num in embedding]
        embed[word] = word_embed
lengths = len(embed['the'])
unk = []
for i in range(lengths):
    unk.append(0.0)

word2idx = {}
idx2word = {}

Embeddings = []

with open('./vocab/f30k_precomp_vocab.json') as f:
    d = json.load(f)
    word2idx = d['word2idx']
    idx2word = d['idx2word']
    keys = word2idx.keys()

    for i in keys:
        if i not in embed.keys():
            Embeddings.append(unk)
        else:
            Embeddings.append(embed[i])

Embeddings = np.array(Embeddings)
print(Embeddings.shape)
np.savetxt('f30k300.txt', Embeddings)

# class l1(object):
#     def __init__(self):
#         c = np.loadtxt('f30k.txt')
#         self.word_embeds = nn.Embedding(8480, 50)
#         self.word_embeds.weight.data.copy_(torch.from_numpy(c))
#         print(self.word_embeds.weight.data)
#
#     def forward(self, x):
#         x = self.word_embeds(x)
#         print(x)
#
#
#
# test = l1()
# t = torch.tensor(np.array([5,6,7,8]))
# test(t)



# c = np.loadtxt('f30k.txt')
# print(type(c))
# embedding = torch.nn.Embedding(8480, 50)















