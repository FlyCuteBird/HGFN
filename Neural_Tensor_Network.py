import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
import argparse

class ImageRelationNTN(nn.Module):

    def __init__(self, image_K, embed_size, bias=True):
        super(ImageRelationNTN, self).__init__()
        # K represents the number of slice
        self.K = image_K
        self.embed = embed_size
        self.K_weights = nn.ModuleList([nn.Linear(self.embed, self.embed, bias=bias) for i in range(self.K)])
        self.fc_V1 = nn.Linear(self.embed, self.K)
        self.fc_V2 = nn.Linear(self.embed, self.K)
        self.fc_U = nn.Linear(self.K, 1)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        # init fc_V2
        r_v1 = np.sqrt(6.) / np.sqrt(self.fc_V1.in_features + self.fc_V1.out_features)
        self.fc_V1.weight.data.uniform_(-r_v1, r_v1)
        self.fc_V1.bias.data.fill_(0)

        # init fc_V2
        r_v2 = np.sqrt(6.) / np.sqrt(self.fc_V2.in_features + self.fc_V2.out_features)
        self.fc_V2.weight.data.uniform_(-r_v2, r_v2)
        self.fc_V2.bias.data.fill_(0)

        # init fc_U
        r_u = np.sqrt(6.) / np.sqrt(self.fc_U.in_features + self.fc_U.out_features)
        self.fc_U.weight.data.uniform_(-r_u, r_u)
        self.fc_U.bias.data.fill_(0)

        # init K_weights
        r_k = np.sqrt(6.) / np.sqrt(self.K_weights[0].in_features + self.K_weights[0].out_features)
        for i in range(self.K):
            self.K_weights[i].weight.data.uniform_(-r_k, r_k)
            self.K_weights[i].bias.data.fill_(0)


    def call_one_image(self, images):
        '''
        :param image: batch_size*36*dim,
        :return: batch_size*36*36 sim of each pair of features
        '''
        batch_size = images.size(0)
        num_feature = images.size(1)
        relation_result = []
        # one_image_relation---->[36, 36]
        for i in range(batch_size):
            one_image_relation = self.compute_image_TNT_ralation(images[i, :, :])
            relation_result.append(one_image_relation)

        # return [num_feature, num_feature]
        return torch.cat(relation_result, dim=0).view(batch_size, num_feature, num_feature, -1)

    def compute_image_TNT_ralation(self, image):
        '''
        image:[1, 36, dim]
        '''
        # image:[36, dim]
        image = image.squeeze()
        num = image.size(0)
       # first_part--->[num*num, K]
        first_part = [self.K_weights[i](image) for i in range(self.K)]
        first_part = torch.cat(first_part, dim=0)
        first_part = torch.mm(first_part, image.transpose(0, 1)).contiguous().view(-1, self.K)
        #  second_part--->[]

        #seconf_part1,2 --->[k, 36]
        second_part1 = self.fc_V1(image).transpose(0, 1).contiguous().view(self.K, num, 1)
        second_part2 = self.fc_V2(image).transpose(0, 1).contiguous().view(self.K, 1, num)
        second_prat = (second_part1 + second_part2).view(-1, self.K)
        result = torch.tanh(torch.add(first_part, second_prat))
        #return self.fc_U(result).view(num, -1)
        return torch.softmax(self.fc_U(result).view(num, -1), dim=1)

    def forward(self, images):
        """compute the relation between image pairs."""
        # relation_weights: [batch, number, number]
        image_relation_weights = self.call_one_image(images)
        return image_relation_weights

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(ImageRelationNTN, self).load_state_dict(new_state)



class TextRelationNTN(nn.Module):

    def __init__(self, text_K, embed_size, bias=True):
        super(TextRelationNTN, self).__init__()
        # K represents the number of slice
        self.K1 = text_K
        self.embed1 = embed_size
        self.K1_weights = nn.ModuleList([nn.Linear(self.embed1, self.embed1, bias=bias) for i in range(self.K1)])
        self.fc_V11 = nn.Linear(self.embed1, self.K1)
        self.fc_V21 = nn.Linear(self.embed1, self.K1)
        self.fc_U1 = nn.Linear(self.K1, 1)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        # init fc_V2
        r_v1 = np.sqrt(6.) / np.sqrt(self.fc_V11.in_features + self.fc_V11.out_features)
        self.fc_V11.weight.data.uniform_(-r_v1, r_v1)
        self.fc_V11.bias.data.fill_(0)

        # init fc_V2
        r_v2 = np.sqrt(6.) / np.sqrt(self.fc_V21.in_features + self.fc_V21.out_features)
        self.fc_V21.weight.data.uniform_(-r_v2, r_v2)
        self.fc_V21.bias.data.fill_(0)

        # init fc_U
        r_u = np.sqrt(6.) / np.sqrt(self.fc_U1.in_features + self.fc_U1.out_features)
        self.fc_U1.weight.data.uniform_(-r_u, r_u)
        self.fc_U1.bias.data.fill_(0)

        # init K_weights
        r_k = np.sqrt(6.) / np.sqrt(self.K1_weights[0].in_features + self.K1_weights[0].out_features)
        for i in range(self.K1):
            self.K1_weights[i].weight.data.uniform_(-r_k, r_k)
            self.K1_weights[i].bias.data.fill_(0)


    def call_one_text(self, texts):
        '''
        :param image: batch_size*36*dim,
        :return: batch_size*36*36 sim of each pair of features
        '''
        batch_size = texts.size(0)
        num_feature = texts.size(1)
        relation_result = []
        # one_image_relation---->[36, 36]
        for i in range(batch_size):
            one_text_relation = self.compute_Text_TNT_ralation(texts[i, :, :])
            relation_result.append(one_text_relation)

        # return [num_feature, num_feature]
        return torch.cat(relation_result, dim=0).view(batch_size, num_feature, num_feature, -1)

    def compute_Text_TNT_ralation(self, text):
        '''
        text:[1, num, dim]
        '''
        # text:[num, dim]
        text = text.squeeze()
        num = text.size(0)
       # first_part--->[num*num, K]
        first_part = [self.K1_weights[i](text) for i in range(self.K1)]
        first_part = torch.cat(first_part, dim=0)
        first_part = torch.mm(first_part, text.transpose(0, 1)).contiguous().view(-1, self.K1)
        #  second_part--->[]

        #seconf_part1,2 --->[k, 36]
        second_part1 = self.fc_V11(text).transpose(0, 1).contiguous().view(self.K1, num, 1)
        second_part2 = self.fc_V21(text).transpose(0, 1).contiguous().view(self.K1, 1, num)
        second_prat = (second_part1 + second_part2).view(-1, self.K1)
        result = torch.tanh(torch.add(first_part, second_prat))
        #return self.fc_U(result).view(num, -1)
        return torch.softmax(self.fc_U1(result).view(num, -1), dim=1)

    def forward(self, texts):
        """compute the relation between image pairs."""
        # relation_weights: [batch, number, number]
        text_relation_weights = self.call_one_text(texts)
        return text_relation_weights

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(TextRelationNTN, self).load_state_dict(new_state)








