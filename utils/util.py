from __future__ import division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.parameter import Parameter
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('agg')
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.metrics import precision_score, recall_score
from scipy.optimize import linear_sum_assignment
import random
import os
import argparse
import logging
import os
import sys

def get_schedule(optimizer, args):
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    return exp_lr_scheduler


def setup_logger(name, save_dir, distributed_rank, train=True, ):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, "log.txt" if train else 'log_eval.txt'),
                                mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger



#######################################################
# Evaluate Critiron
#######################################################
def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class BCE(nn.Module):
    eps = 1e-7 # Avoid calculating log(0). Use the small value of float16.
    def forward(self, prob1, prob2, simi):
        # prob1 and prob2 are pair enum
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1)==len(prob2)==len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),str(len(prob2)),str(len(simi)))
        P = prob1.mul_(prob2)
        P = P.sum(1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(BCE.eps).log_()
        return neglogP.mean()

class SoftCE(nn.Module):
    eps = 1e-7 # Avoid calculating log(0). Use the small value of float16.
    def forward(self, prob1, prob2, simi):
        # prob1 and prob2 are pair enum
        # simi: a soft prob value
        assert len(prob1)==len(prob2)==len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),str(len(prob2)),str(len(simi)))
        P = prob1.mul_(prob2)
        P = P.sum(1)
        logP = P.add_(SoftCE.eps).log_()
        negsimlogP = -simi.mul_(logP)
        return negsimlogP.mean()

def PairEnum(x,mask=None):
    # Enumerate all pairs of feature in x
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0),1)
    x2 = x.repeat(1,x.size(0)).view(-1,x.size(1))
    if mask is not None:
        xmask = mask.view(-1,1).repeat(1,x.size(1))
        #dim 0: #sample, dim 1:#feature 
        x1 = x1[xmask].view(-1,x.size(1))
        x2 = x2[xmask].view(-1,x.size(1))
    return x1,x2

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
    

# prob2 == prob_ub
def compute_bce(feat, prob2, prob2_bar, mask_lb, criterion2, args, 
     return_acc=False, label=None):
    device = torch.device("cuda" if args.cuda else "cpu")
    rank_feat = (feat[~mask_lb]).detach()
    rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
    rank_idx1, rank_idx2= PairEnum(rank_idx)
    rank_idx1, rank_idx2=rank_idx1[:, :args.topk], rank_idx2[:, :args.topk]
    rank_idx1, _ = torch.sort(rank_idx1, dim=1)
    rank_idx2, _ = torch.sort(rank_idx2, dim=1)

    rank_diff = rank_idx1 - rank_idx2
    rank_diff = torch.sum(torch.abs(rank_diff), dim=1)
    target_ulb = torch.ones_like(rank_diff).float().to(device) 
    target_ulb[rank_diff>0] = -1 

    if return_acc:
        rlabel = label[~mask_lb].view(-1, 1)
        label1, label2 = PairEnum(rlabel)

        label12_diff = label1 - label2
        label12_diff = torch.sum(torch.abs(label12_diff), dim=1)

        real_target_ulb = torch.ones_like(label12_diff).float().to(device)
        real_target_ulb[label12_diff > 0] = -1
        
        ulb_acc = (real_target_ulb == target_ulb).sum().item() / real_target_ulb.shape[0]
        ################################################################
        # calculate recall and precision 
        y_true = real_target_ulb.detach().clone().cpu().numpy()
        y_true[y_true == -1] = 0
        y_pred = target_ulb.detach().clone().cpu().numpy()
        y_pred[y_pred == -1] = 0
        
        p = precision_score(y_true, y_pred)#, average='macro')
        r = recall_score(y_true, y_pred)#, average='macro')
        ################################################################

    # calc BCE loss using enum
    prob1_ulb, _ = PairEnum(prob2[~mask_lb]) 
    _, prob2_ulb = PairEnum(prob2_bar[~mask_lb]) 
    loss_bce = criterion2(prob1_ulb, prob2_ulb, target_ulb)
    if return_acc:
        return loss_bce, ulb_acc, p, r
    return loss_bce

def compute_part_bce(feat, prob2, prob2_bar, mask_lb, criterion2, args, return_acc=False, label=None):
    device = torch.device("cuda" if args.cuda else "cpu")
    com_feat = (feat[~mask_lb]).detach()

    rank_idx = torch.argsort(com_feat, dim=1, descending=True)
    rank_idx1, rank_idx2 = PairEnum(rank_idx)
    rank_idx1, rank_idx2 = rank_idx1[:, :args.topk], rank_idx2[:, :args.topk]
    rank_idx1, _ = torch.sort(rank_idx1, dim=1)
    rank_idx2, _ = torch.sort(rank_idx2, dim=1)

    # using diff to check if topk is the same
    # == 0 is the same
    rank_diff = rank_idx1 - rank_idx2
    rank_diff = torch.sum(torch.abs(rank_diff), dim=1)
    target_ulb = torch.ones_like(rank_diff).float().to(device) 
    target_ulb[rank_diff>0] = -1 
    
    if return_acc:
        rlabel = label[~mask_lb].view(-1, 1)
        label1, label2 = PairEnum(rlabel)

        label12_diff = label1 - label2
        label12_diff = torch.sum(torch.abs(label12_diff), dim=1)

        real_target_ulb = torch.ones_like(label12_diff).float().to(device)
        real_target_ulb[label12_diff > 0] = -1
        
        ulb_acc = (real_target_ulb == target_ulb).sum().item() / real_target_ulb.shape[0]
        ################################################################
        # calculate recall and precision 
        y_true = real_target_ulb.detach().clone().cpu().numpy()
        y_true[y_true == -1] = 0
        y_pred = target_ulb.detach().clone().cpu().numpy()
        y_pred[y_pred == -1] = 0
        
        p = precision_score(y_true, y_pred)#, average='macro')
        r = recall_score(y_true, y_pred)#, average='macro')
        ################################################################

    # calc BCE loss using enum
    prob1_ulb, _ = PairEnum(prob2[~mask_lb]) 
    _, prob2_ulb = PairEnum(prob2_bar[~mask_lb]) 
    loss_bce = criterion2(prob1_ulb, prob2_ulb, target_ulb)
    if return_acc:
        return loss_bce, ulb_acc, p, r
    return loss_bce
    

kl_crit = nn.KLDivLoss(reduction='batchmean')
def symmetric_kld(p, q):
    # p and q are logits
    skld = kl_crit(F.log_softmax(p, 1), F.softmax(q, 1))
    skld += kl_crit(F.log_softmax(q, 1), F.softmax(p, 1))
    return skld / 2







