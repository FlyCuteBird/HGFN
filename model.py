
""" HGMN model"""

import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
from graph_model import VisualGraph, TextualGraph
from bert_embedding import Bert_Embeddings


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


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):

        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)


# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, word_dim, embed_size, num_layers,
                 use_bi_gru=False, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # bert embedding
        self.embedd = Bert_Embeddings()

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.rnn = nn.GRU(word_dim, embed_size, num_layers,
                          batch_first=True, bidirectional=use_bi_gru)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # bert_embedding
        x = self.embedd(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        if self.use_bi_gru:
            cap_emb = (cap_emb[:, :, :cap_emb.size(2) / 2] +
                       cap_emb[:, :, cap_emb.size(2) / 2:]) / 2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb, cap_len


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, opt, margin=0):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.Rank_Loss = opt.Rank_Loss

    def forward(self, scores):
        # compute image-sentence score matrix
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the DynamciTopK, maximum or all violating negative for each query

        if self.Rank_Loss == 'DynamicTopK_Negative':
            topK = int((cost_s > 0.).sum() / (cost_s.size(0) + 0.00001) + 1)
            cost_s, index1 = torch.sort(cost_s, descending=True, dim=-1)
            cost_im, index2 = torch.sort(cost_im, descending=True, dim=0)

            return cost_s[:, 0:topK].sum() + cost_im[0:topK, :].sum()

        elif self.Rank_Loss == 'Hardest_Negative':
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

            return cost_s.sum() + cost_im.sum()

        else:
            return cost_s.sum() + cost_im.sum()


class HGMN(object):
    """
    Heterogeneous Graph Matching Network for Image-text Matching （HGMN）
    """

    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImagePrecomp(
            opt.img_dim, opt.embed_size, opt.no_imgnorm)
        self.txt_enc = EncoderText(opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   use_bi_gru=opt.bi_gru,
                                   no_txtnorm=opt.no_txtnorm)
        self.i2t_match_G = VisualGraph(
            opt.feat_dim, opt.hid_dim, opt.out_dim, opt.region_relation, opt.image_K, opt.embed_size, dropout=.5)
        self.t2i_match_G = TextualGraph(
            opt.feat_dim, opt.hid_dim, opt.out_dim, opt.text_K, opt.embed_size, dropout=.5)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.i2t_match_G.cuda()
            self.t2i_match_G.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(opt=opt, margin=opt.margin)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.fc.parameters())
        params += list(self.i2t_match_G.parameters())
        params += list(self.t2i_match_G.parameters())

        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0
        self.opt = opt

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(),
                      self.i2t_match_G.state_dict(),
                      self.t2i_match_G.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])
        self.i2t_match_G.load_state_dict(state_dict[2])
        self.t2i_match_G.load_state_dict(state_dict[3])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()
        self.i2t_match_G.train()
        self.t2i_match_G.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()
        self.i2t_match_G.eval()
        self.t2i_match_G.eval()

    def forward_emb(self, images, captions, lengths, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images, volatile=volatile)
        captions = Variable(captions, volatile=volatile)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward
        img_emb = self.img_enc(images)

        # cap_emb (tensor), cap_lens (list)
        cap_emb, cap_lens = self.txt_enc(captions, lengths)
        return img_emb, cap_emb, cap_lens

    def forward_sim(self, img_emb, cap_emb, bbox,  cap_lens):
        i2t_scores = self.i2t_match_G(
            img_emb, cap_emb, bbox, cap_lens, self.opt)
        t2i_scores = self.t2i_match_G(
            img_emb, cap_emb,  cap_lens, self.opt)
        scores = i2t_scores + t2i_scores
       # split or joint (T2V , V2T)
       # scores = t2i_scores
       # scores = i2t_scores
        return scores

    def forward_loss(self, scores, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(scores)
        self.logger.update('Le', loss.item())
        return loss

    def train_emb(self, images, captions, bboxes, lengths, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb, cap_lens = self.forward_emb(
            images, captions, lengths)

        scores = self.forward_sim(img_emb, cap_emb, bboxes, cap_lens)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(scores)
        
        # # log Loss
        # file = open(self.opt.model_name + '/' + self.opt.region_relation + '/' + '%s_%s_Loss.txt' %(self.opt.region_relation, self.opt.windows_size), 'a' )
        # file.write(str(self.Eiters) + "    " + str(loss.item()) + "\n")
        # file.close()
        
        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()
