"""GraphConvolution Layer"""

import torch
import numpy as np

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F


class t2iGraphConvolution(Module):

    def __init__(self,
                 in_feat_dim,
                 out_feat_dim,
                 n_kernels,
                 coordinate_dim,
                 region_relation,
                 bias=False):
        super(t2iGraphConvolution, self).__init__()


        # Set parameters
        self.n_kernels = n_kernels
        self.coordinate_dim = coordinate_dim
        self.in_feat_dim = in_feat_dim
        self.out_feat_dim = out_feat_dim
        self.bias = bias

        self.region_relation = region_relation

        # Convolution filters weights
        self.conv_weights = nn.ModuleList([nn.Linear(
            in_feat_dim, out_feat_dim // n_kernels, bias=bias) for i in range(n_kernels)])

        # Parameters of the Gaussian kernels
        self.mean_rho = Parameter(torch.Tensor(n_kernels, 1))
        self.mean_theta = Parameter(torch.Tensor(n_kernels, 1))
        self.precision_rho = Parameter(torch.Tensor(n_kernels, 1))
        self.precision_theta = Parameter(torch.Tensor(n_kernels, 1))
        # qin
        self.mean_NTN_relation = Parameter(torch.Tensor(n_kernels, 1))
        self.precision_NTN_relation = Parameter(torch.Tensor(n_kernels, 1))

        self.init_parameters()

    def init_parameters(self):
        # Initialise Gaussian parameters

        # position relation
        self.mean_theta.data.uniform_(-np.pi, np.pi)
        self.mean_rho.data.uniform_(0, 1.0)
        self.precision_theta.data.uniform_(0.0, 1.0)
        self.precision_rho.data.uniform_(0.0, 1.0)

        # NTN relation
        self.mean_NTN_relation.data.uniform_(0, 1.0)
        self.precision_NTN_relation.data.uniform_(0.0, 1.0)

    def forward(self, neighbourhood_features, region_relation_weights):

        # set parameters
        batch_size = neighbourhood_features.size(0)
        K = neighbourhood_features.size(1)
        neighbourhood_size = neighbourhood_features.size(2)

        # choose the way to compute  relation of image regions
        if self.region_relation == 'Position_relation':
            weights = self.get_Postion_gaussian_weights(region_relation_weights)
        else:
            weights = self.get_NTN_gaussian_weights(region_relation_weights)

        weights = weights.view(
            batch_size * K, neighbourhood_size, self.n_kernels)

        # compute convolved features
        neighbourhood_features = neighbourhood_features.view(
            batch_size * K, neighbourhood_size, -1)
        convolved_features = self.convolution(neighbourhood_features, weights)
        convolved_features = convolved_features.view(-1, K, self.out_feat_dim)

        return convolved_features

    def get_Postion_gaussian_weights(self, region_relation_weights):

        # compute rho weights
        diff = (region_relation_weights[:, :, :, 0].contiguous(
        ).view(-1, 1) - self.mean_rho.view(1, -1))**2
        weights_rho = torch.exp(-0.5 * diff /
                                (1e-14 + self.precision_rho.view(1, -1)**2))

        # compute theta weights
        first_angle = torch.abs(region_relation_weights[:, :, :, 1].contiguous(
        ).view(-1, 1) - self.mean_theta.view(1, -1))
        second_angle = torch.abs(2 * np.pi - first_angle)
        weights_theta = torch.exp(-0.5 * (torch.min(first_angle, second_angle)**2)
                                  / (1e-14 + self.precision_theta.view(1, -1)**2))

        weights = weights_rho * weights_theta

        weights[(weights != weights).detach()] = 0

        # normalise weights
        weights = weights / torch.sum(weights, dim=1, keepdim=True)

        return weights

    def get_NTN_gaussian_weights(self, region_relation_weights):

        diff = (region_relation_weights[:, :, :, 2].contiguous(
        ).view(-1, 1) - self.mean_NTN_relation.view(1, -1))**2
        weights_NTN = torch.exp(-0.5 * diff /
                                (1e-14 + self.precision_NTN_relation.view(1, -1)**2))

        # normalise weights
        weights_NTN = weights_NTN / torch.sum(weights_NTN, dim=1, keepdim=True)

        return weights_NTN


    def convolution(self, neighbourhood, weights):

        weighted_neighbourhood = torch.bmm(
            weights.transpose(1, 2), neighbourhood)

        # convolutions
        weighted_neighbourhood = [self.conv_weights[i](
            weighted_neighbourhood[:, i]) for i in range(self.n_kernels)]
        convolved_features = torch.cat(
            [i.unsqueeze(1) for i in weighted_neighbourhood], dim=1)
        convolved_features = convolved_features.view(-1, self.out_feat_dim)

        return convolved_features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(t2iGraphConvolution, self).load_state_dict(new_state)


class i2tGraphConvolution(Module):

    def __init__(self,
                 in_feat_dim,
                 out_feat_dim,
                 n_kernels,
                 bias=False):
        super(i2tGraphConvolution, self).__init__()

        # Set parameters
        self.n_kernels = n_kernels
        self.in_feat_dim = in_feat_dim
        self.out_feat_dim = out_feat_dim
        self.bias = bias

        # Convolution filters weights
        self.conv_weights = nn.ModuleList([nn.Linear(
            in_feat_dim, out_feat_dim // n_kernels, bias=bias) for i in range(n_kernels)])

        # Parameters of the Gaussian kernels
        self.params = Parameter(torch.Tensor(n_kernels, 1))

        self.init_parameters()

    def init_parameters(self):
        # Initialise Gaussian parameters
        self.params.data.uniform_(-1.0, 1.0)

    def forward(self, neighbourhood_features, neighbourhood_weights):

        # set parameters
        batch_size = neighbourhood_features.size(0)
        K = neighbourhood_features.size(1)
        neighbourhood_size = neighbourhood_features.size(2)

        # compute neighborhood kernel weights
        weights = self.compute_weights(neighbourhood_weights)
        weights = weights.view(
            batch_size * K, neighbourhood_size, self.n_kernels)

        # compute convolved features
        neighbourhood_features = neighbourhood_features.view(
            batch_size * K, neighbourhood_size, -1)
        convolved_features = self.convolution(neighbourhood_features, weights)
        convolved_features = convolved_features.view(-1, K, self.out_feat_dim)

        return convolved_features

    def compute_weights(self, neighbourhood_weights):

        batch_size = neighbourhood_weights.size(0)
        n_word = neighbourhood_weights.size(1)

        weights = neighbourhood_weights.view(-1, 1) * self.params.view(1, -1)
        weights = weights.view(batch_size * n_word, n_word, -1)
        # normalise weights
        weights = weights / torch.sum(weights, dim=1, keepdim=True)
        return weights

    def convolution(self, neighbourhood, weights):

        weighted_neighbourhood = torch.bmm(
            weights.transpose(1, 2), neighbourhood)

        # convolutions
        weighted_neighbourhood = [self.conv_weights[i](
            weighted_neighbourhood[:, i]) for i in range(self.n_kernels)]
        convolved_features = torch.cat(
            [i.unsqueeze(1) for i in weighted_neighbourhood], dim=1)
        convolved_features = convolved_features.view(-1, self.out_feat_dim)

        return convolved_features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(i2tGraphConvolution, self).load_state_dict(new_state)
