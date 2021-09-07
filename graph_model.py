from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from Convolution_Layers import t2iGraphConvolution, i2tGraphConvolution
from Neural_Tensor_Network import ImageRelationNTN, TextRelationNTN
import numpy as np

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def inter_relations(K, Q, xlambda):
    """
    Q: (batch, queryL, d)
    K: (batch, sourceL, d)
    return (batch, queryL, sourceL)
    """
    batch_size, queryL = Q.size(0), Q.size(1)
    batch_size, sourceL = K.size(0), K.size(1)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    queryT = torch.transpose(Q, 1, 2)

    attn = torch.bmm(K, queryT)
    
    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.Softmax(dim=1)(attn * xlambda)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)

    return attn


class VisualGraph(nn.Module):

    def __init__(self,
                 feat_dim,
                 hid_dim,
                 out_dim,
                 region_relation,
                 image_K,
                 embed_size,
                 dropout,
                 n_kernels=8):
        '''
        ## Variables:
        - feat_dim: dimensionality of input image features
        - out_dim: dimensionality of the output
        - dropout: dropout probability
        - n_kernels : number of Gaussian kernels for convolutions
        - bias: whether to add a bias to Gaussian kernels
        '''

        super(VisualGraph, self).__init__()

        # Set parameters
        self.feat_dim = feat_dim
        self.out_dim = out_dim

        # Set the type of region relation
        self.region_relation = region_relation

        # graph convolution layers
        self.graph_convolution_1 = \
            t2iGraphConvolution(feat_dim, hid_dim, n_kernels, 2, region_relation)

        # NTN relation weights
        self.NTN_relation_weights = ImageRelationNTN(image_K, embed_size)

        # # output classifier
        self.out_1 = nn.utils.weight_norm(nn.Linear(hid_dim, hid_dim))
        self.out_2 = nn.utils.weight_norm(nn.Linear(hid_dim, out_dim))

    def Text2Image_fusion(self, tnodes, vnodes, n_block, xlambda):

        inter_relation = inter_relations(tnodes, vnodes, xlambda)
        # Compute sim with weighted context
        # (batch, n_word, n_region)
        attnT = torch.transpose(inter_relation, 1, 2)
        contextT = torch.transpose(tnodes, 1, 2)  # (batch, dim, n_word)
        weightedContext = torch.bmm(contextT, attnT)  # (batch, dim, n_region)
        weightedContextT = torch.transpose(
            weightedContext, 1, 2)  # (batch, n_region, dims)

        # Multi-segments similarity
        qry_set = torch.split(vnodes, n_block, dim=2)
        ctx_set = torch.split(weightedContextT, n_block, dim=2)
       
        qry_set = torch.stack(qry_set, dim=2)
        ctx_set = torch.stack(ctx_set, dim=2)

        # (batch, n_region, num_block)
        vnode_mvector = cosine_similarity(
            qry_set, ctx_set, dim=-1)

        return vnode_mvector

    def Image2Text_score(self, vnode_mvector, relation_weights):
        # (batch, n_region, n_region, num_block)
        batch, n_region = vnode_mvector.size(0), vnode_mvector.size(1)

        neighbor_image = vnode_mvector.unsqueeze(
            2).repeat(1, 1, n_region, 1)

        # aggregate neighborhood information
        hidden_graph = self.graph_convolution_1(neighbor_image, relation_weights)
        hidden_graph = hidden_graph.view(batch * n_region, -1)

        # infer matching score
        sim = self.out_2(self.out_1(hidden_graph).tanh())
        sim = sim.view(batch, -1).mean(dim=1, keepdim=True)

        return sim

    def forward(self, images, captions, bbox, cap_lens, opt):
        # bbox-->(batch_size, K, 4), k representations image regions or text words
        similarities = []
        n_block = opt.embed_size // opt.num_block
        n_image, n_caption = images.size(0), captions.size(0)

        bb_size = (bbox[:, :, 2:] - bbox[:, :, :2])
        bb_centre = bbox[:, :, :2] + 0.5 * bb_size
        # position weights
        position_relation_weights = self.compute_pseudo(bb_centre).cuda()

        # NTN weights
        NTN_relation_weights = self.NTN_relation_weights(images).cuda()
        relation_weights = torch.cat((position_relation_weights, NTN_relation_weights), dim=-1)
        for i in range(n_caption):
            # Get the i-th text description
            n_word = cap_lens[i]
            cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
            # --> (n_image, n_word, d)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)

            # --> compute similarity between query region and context word
            t2i_fusion = self.Text2Image_fusion(
                cap_i_expand, images, n_block, opt.lambda_softmax)
            i2t_score = self.Image2Text_score(
                t2i_fusion, relation_weights)

            similarities.append(i2t_score)

        similarities = torch.cat(similarities, 1)
        return similarities

    def compute_pseudo(self, bb_centre):
        '''

        Computes pseudo-coordinates from bounding box centre coordinates

        '''
        K = bb_centre.size(1)

        # Compute cartesian coordinates (batch_size, K, K, 2)
        pseudo_coord = bb_centre.view(-1, K, 1, 2) - \
            bb_centre.view(-1, 1, K, 2)

        # Conver to polar coordinates
        rho = torch.sqrt(
            pseudo_coord[:, :, :, 0]**2 + pseudo_coord[:, :, :, 1]**2)
        theta = torch.atan2(
            pseudo_coord[:, :, :, 0], pseudo_coord[:, :, :, 1])
        pseudo_coord = torch.cat(
            (torch.unsqueeze(rho, -1), torch.unsqueeze(theta, -1)), dim=-1)
        # pseudo_coord-->(batch_size, k, k, 2) k represents regions(36), 2-->rho, theta; rho is the radian value.
        return pseudo_coord


class TextualGraph(nn.Module):

    def __init__(self,
                 feat_dim,
                 hid_dim,
                 out_dim,
                 text_K,
                 embed_size,
                 dropout,
                 n_kernels=8):

        super(TextualGraph, self).__init__()

        # Set parameters
        self.feat_dim = feat_dim
        self.out_dim = out_dim

        self.graph_convolution_3 = \
            i2tGraphConvolution(feat_dim, hid_dim, n_kernels, 2)

        # text word NTN relation
        self.text_NTN_relation_weights = TextRelationNTN(text_K, embed_size)

        # # output classifier
        self.out_1 = nn.utils.weight_norm(nn.Linear(hid_dim, hid_dim))
        self.out_2 = nn.utils.weight_norm(nn.Linear(hid_dim, out_dim))


    def build_neighborhood_graph(self, windows_size, lens):

        adj = np.zeros((lens, lens), dtype=np.int)
        for i in range(lens):
            for j in range(lens):
                if abs(i-j) <= windows_size:
                    adj[i, j] = 1
        return torch.from_numpy(adj).cuda().float()

    def Image2Text_fusion(self, vnodes, tnodes, n_block, xlambda):

        inter_relation = inter_relations(vnodes, tnodes, xlambda)

        # Compute sim with weighted context
        attnT = torch.transpose(inter_relation, 1, 2)
        contextT = torch.transpose(vnodes, 1, 2)
        weightedContext = torch.bmm(contextT, attnT)
        weightedContextT = torch.transpose(
            weightedContext, 1, 2)

        # Multi-segments similarity
        qry_set = torch.split(tnodes, n_block, dim=2)
        ctx_set = torch.split(weightedContextT, n_block, dim=2)

        qry_set = torch.stack(qry_set, dim=2)
        ctx_set = torch.stack(ctx_set, dim=2)

        tnode_mvector = cosine_similarity(
            qry_set, ctx_set, dim=-1)
        return tnode_mvector

    def Text2Image_score(self, tnode_mvector, text_TNT_relation, opt):
        # (batch, n_word, 1, num_block)
        tnode_mvector = tnode_mvector.unsqueeze(2)
        batch, n_word = tnode_mvector.size(0), tnode_mvector.size(1)

        if opt.text_graph == 'connected_graph':

            neighbor_nodes = tnode_mvector.repeat(1, 1, n_word, 1)
            neighbor_weights = text_TNT_relation.repeat(batch, 1, 1, 1)

        elif opt.text_graph == 'neighborhood_graph':
            # Build adjacency matrix for each text query
            # (1, n_word, n_word, 1)
            adj_mtx = self.build_neighborhood_graph(opt.windows_size, n_word)
            adj_mtx = adj_mtx.view(n_word, n_word).unsqueeze(0).unsqueeze(-1)
            # (batch, n_word, n_word, num_block)
            neighbor_nodes = adj_mtx * tnode_mvector
            # (batch, n_word, n_word, 1)
            neighbor_weights = l2norm(adj_mtx * text_TNT_relation, dim=2)
            neighbor_weights = neighbor_weights.repeat(batch, 1, 1, 1)

        hidden_graph = self.graph_convolution_3(
            neighbor_nodes, neighbor_weights)
        hidden_graph = hidden_graph.view(batch * n_word, -1)

        sim = self.out_2(self.out_1(hidden_graph).tanh())
        sim = sim.view(batch, -1).mean(dim=1, keepdim=True)
        return sim

    def forward(self, images, captions, cap_lens, opt):
        n_image = images.size(0)
        n_caption = captions.size(0)
        similarities = []
        n_block = opt.embed_size // opt.num_block
        for i in range(n_caption):
            # Get the i-th text description
            n_word = cap_lens[i]
            cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()

            # --> (n_image, n_word, d)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)

            text_TNT_relation = self.text_NTN_relation_weights(cap_i)
            i2t_fusion = self.Image2Text_fusion(
                images, cap_i_expand, n_block, opt.lambda_softmax)
            t2i_score = self.Text2Image_score(
                i2t_fusion, text_TNT_relation, opt)

            similarities.append(t2i_score)

        # (n_image, n_caption)
        similarities = torch.cat(similarities, 1)

        return similarities
