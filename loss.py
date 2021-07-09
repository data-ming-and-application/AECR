"""
# Pytorch implementation for paper
# "AECR: Alignment Efficient Cross-Modal Retrieval Considering Transferable Representation Learning"
# Yang Yang, Jinyi Guo, Hengshu Zhu, Dianhai Yu, Fuzhen Zhuang, Hui Xiong and Jian Yang
"""


import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

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

#Semantic loss
class Intra_modal(nn.Module):
    def __init__(self):
        super(Intra_modal,self).__init__()

    def domain_adversarial_loss_d(self,n_label,p_label):
        da_loss_d=-torch.log(p_label).mean()-torch.log(1-n_label).mean()
        return da_loss_d

    def domain_adversarial_loss_e(self,n_label,p_label):
        da_loss_e=-torch.log(1-p_label).mean()-torch.log(n_label).mean()
        return da_loss_e

    def loss_d(self,n_label,p_label,n1_label,p1_label):
        lss_d=self.domain_adversarial_loss_d(n_label,p_label)+self.domain_adversarial_loss_d(n1_label,p1_label)
        return lss_d

    def loss_e(self,n_label,p_label,n1_label,p1_label):
        lss_e=self.domain_adversarial_loss_e(n_label,p_label)+self.domain_adversarial_loss_e(n1_label,p1_label)
        return lss_e

def get_sim(raw_im, raw_s, eps=1e-8):
    raw_s_norm=torch.norm(raw_s,2,2).clamp(min=eps).unsqueeze(2)
    #raw_s_norm = torch.pow(raw_s, 2).sum(dim=2, keepdim=True).sqrt() + eps
    raw_s_norm = torch.transpose(raw_s_norm, 1, 2)
    raw_s = torch.transpose(raw_s, 1, 2)
    similarity=[]
    for i in range(raw_im.shape[0]):
        im = raw_im[i]
        im = im.view(-1, im.shape[0], im.shape[1])
        im_norm=torch.norm(im,2,2).clamp(min=eps).unsqueeze(2)
        norm = torch.matmul(im_norm, raw_s_norm)
        region_scores = torch.matmul(im, raw_s)
        region_scores = torch.div(region_scores, norm)
        max_scores, _ = torch.max(region_scores, dim=1, keepdim=True)
        temp_sim = max_scores.sum(dim=2)
        temp_sim = torch.transpose(temp_sim, 0, 1)
        similarity.append(temp_sim)
    similarity=torch.cat(similarity,dim=0).cuda()
    return similarity

# Structure loss
class Inter_modal(nn.Module):
    def __init__(self,tau=3,K=3):
        super(Inter_modal,self).__init__()
        self.tau=tau
        self.K=K

    def semantic_transfer_loss(self,m1_n_emb,m1_p_emb,m2_p_emb):
        #similarity between modal1 and modal2
        m1n2m1p_sim=get_sim(m1_n_emb,m1_p_emb)
        m1_k_sim = []
        m2_k_sim = []
        for i in range(m1n2m1p_sim.size(0)):
            # choose first k similar images
            m1_i_sim,index=torch.sort(m1n2m1p_sim[i],descending=True)
            m1_i_k_sim=m1_i_sim[:self.K].unsqueeze(0)
            # choose first k captions correlated to k images
            m2_i_k_emb=m2_p_emb[index[:self.K]]
            m2_i_k_sim=get_sim(m1_n_emb[i].unsqueeze(0),m2_i_k_emb)
            m1_k_sim.append(m1_i_k_sim)
            m2_k_sim.append(m2_i_k_sim)
        m1_k_sim=torch.cat(m1_k_sim,dim=0).cuda()
        m2_k_sim=torch.cat(m2_k_sim,dim=0).cuda()
        # KL divergence
        m2_k_log_soft=F.log_softmax(torch.div(m2_k_sim,self.tau),dim=-1)
        m1_k_softmax=F.softmax(torch.div(m1_k_sim,self.tau),dim=-1)
        kl_sum=F.kl_div(m2_k_log_soft,m1_k_softmax,reduction='sum')
        return kl_sum

    def forward(self,img_n_emb, cap_n_emb,img_p_emb, cap_p_emb):
        loss=self.semantic_transfer_loss(img_n_emb,img_p_emb,cap_p_emb)+\
             self.semantic_transfer_loss(cap_n_emb,cap_p_emb,img_p_emb)
        return loss


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation

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
        if torch.cuda.is_available():
            I = mask.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()
