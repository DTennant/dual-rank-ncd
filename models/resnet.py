from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import  transforms
import numpy as np
from models.part_dict import PartDict
from torchvision.models import resnet50
from copy import deepcopy

class MoCo_R50(nn.Module):
    def __init__(self, num_labeled_classes=5, num_unlabeled_classes=5, moco_path='', test_use='g'):
        super().__init__()
        self.net = resnet50(False)
        self.head1_g = nn.Linear(2048, num_labeled_classes)
        self.head2_g = nn.Linear(2048, num_unlabeled_classes)

        self.head1_p = nn.Linear(2048, num_labeled_classes)
        self.head2_p = nn.Linear(2048, num_unlabeled_classes)

        assert moco_path != ''
        state = torch.load(moco_path, map_location='cpu')['state_dict']
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in state.items():
            if not k.startswith("module.encoder_q."):
                continue
            k = k[len('module.encoder_q.'):]
            new_dict.update({k : v})
        print(self.net.load_state_dict(new_dict, strict=False))
        
        self.layer4_g = self.net.layer4
        self.layer4_p = deepcopy(self.net.layer4)
        del self.net.layer4

        # for contrastive
        self.K = 2048
        self.T = 0.07
        self.n_dim = self.net.fc.in_features

        self.index_g = 0
        self.register_buffer('memory_g', torch.randn(self.K, self.n_dim))
        self.memory_g = F.normalize(self.memory_g)

        self.index_p = 0
        self.register_buffer('memory_p', torch.randn(self.K, self.n_dim))
        self.memory_p = F.normalize(self.memory_p)
        
        self.part_dict = PartDict(2048, self.n_dim)
        
        self.test_use = test_use
        

    def update_pointer_g(self, bsz):
        self.index_g = (self.index_g + bsz) % self.K

    def update_memory_g(self, k, queue):
        with torch.no_grad():
            num_neg = k.shape[0]
            out_ids = torch.arange(num_neg).cuda()
            out_ids = torch.fmod(out_ids + self.index_g, self.K).long()
            queue.index_copy_(0, out_ids, k)

    def update_pointer_p(self, bsz):
        self.index_p = (self.index_p + bsz) % self.K

    def update_memory_p(self, k, queue):
        with torch.no_grad():
            num_neg = k.shape[0]
            out_ids = torch.arange(num_neg).cuda()
            out_ids = torch.fmod(out_ids + self.index_p, self.K).long()
            queue.index_copy_(0, out_ids, k)

    def compute_logits(self, q, k, queue):
        bsz = q.shape[0]
        pos = torch.bmm(q.view(bsz, 1, -1), k.view(bsz, -1, 1))
        pos = pos.view(bsz, 1)

        neg = torch.mm(queue, q.transpose(1, 0))
        neg = neg.transpose(0, 1)
        
        out = torch.cat((pos, neg), dim=1)
        out = torch.div(out, self.T)
        out = out.squeeze().contiguous()

        return out
        
    def forward(self, x, use_ranking=True):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        
        f_g = self.layer4_g(x)

        f_g = self.net.avgpool(f_g)
        f_g = torch.flatten(f_g, 1)
        if use_ranking:
            f_g = F.relu(f_g)
        lb_logit_g = self.head1_g(f_g)
        ul_logit_g = self.head2_g(f_g)


        f_p = self.layer4_p(x)
        fm_p = f_p

        f_p = self.net.avgpool(f_p)
        f_p = torch.flatten(f_p, 1)
        if use_ranking:
            f_p = F.relu(f_p)
        lb_logit_p = self.head1_p(f_g)
        ul_logit_p = self.head2_p(f_g)
        
        if not self.training:
            if self.test_use == 'p':
                return lb_logit_p, ul_logit_p, f_p
            return lb_logit_g, ul_logit_g, f_g

        # return x1, x2, x
        return {
            'output': [
                [lb_logit_g, ul_logit_g, f_g], 
                [lb_logit_p, ul_logit_p, f_p]
            ], 
            'out_featmap': [fm_p]
        }
        

    

