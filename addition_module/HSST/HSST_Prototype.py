""" 
@author: Hang Du, Jun Wang 
@date: 20211020
@contact: jun21wangustc@gmail.com   
""" 

import torch
from torch.nn import Module
import math
import random
import torch.nn.functional as F

class HSST_Prototype(Module):
    """Implementation for "Semi-Siamese Training for Shallow Face Learning".
    """
    def __init__(self, feat_dim=512, queue_size=256, scale=30.0, loss_type='softmax', margin=0.0):
        super(HSST_Prototype, self).__init__()
        self.queue_size = queue_size
        self.feat_dim = feat_dim
        self.scale = scale
        self.margin = margin
        self.loss_type = loss_type
        # initialize the nir and vis prototype queue
        self.register_buffer('vis_queue', torch.rand(feat_dim,queue_size).uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5))
        self.vis_queue = F.normalize(self.vis_queue, p=2, dim=0) 
        self.register_buffer('nir_queue', torch.rand(feat_dim,queue_size).uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5))
        self.nir_queue = F.normalize(self.nir_queue, p=2, dim=0) 
        self.index = 0
        self.label_list = [-1] * queue_size

    def add_margin(self, cos_theta, label, batch_size):
        cos_theta = cos_theta.clamp(-1, 1) 
        # additive cosine margin
        if self.loss_type == 'am_softmax':
            cos_theta_m = cos_theta[torch.arange(0, batch_size), label].view(-1, 1) - self.margin
            cos_theta.scatter_(1, label.data.view(-1, 1), cos_theta_m)
        # additive angurlar margin
        elif self.loss_type == 'arc_softmax':
            gt = cos_theta[torch.arange(0, batch_size), label].view(-1, 1)
            sin_theta = torch.sqrt(1.0 - torch.pow(gt, 2))
            cos_theta_m = gt * math.cos(self.margin) - sin_theta * math.sin(self.margin) 
            cos_theta.scatter_(1, label.data.view(-1, 1), cos_theta_m)
        return cos_theta

    def compute_theta(self, p, g, label, batch_size, VIS_Prototype=True):
        if VIS_Prototype:
            vis_queue = self.vis_queue.clone()
            vis_queue[:,self.index:self.index+batch_size] = g.transpose(0,1)
            cos_theta = torch.mm(p, vis_queue.detach())
        else:
            nir_queue = self.nir_queue.clone()
            nir_queue[:,self.index:self.index+batch_size] = g.transpose(0,1)
            cos_theta = torch.mm(p, nir_queue.detach())            
        cos_theta = self.add_margin(cos_theta, label, batch_size)
        return cos_theta

    def update_queue(self, vis_g, nir_g, cur_ids, batch_size):
        with torch.no_grad():
            self.vis_queue[:,self.index:self.index+batch_size] = vis_g.transpose(0,1)
            self.nir_queue[:,self.index:self.index+batch_size] = nir_g.transpose(0,1)
            for image_id in range(batch_size):
                self.label_list[self.index + image_id] = cur_ids[image_id].item()
            self.index = (self.index + batch_size) % self.queue_size

    def get_id_set(self):
        id_set = set()
        for label in self.label_list:
            if label != -1:
                id_set.add(label)
        return id_set

    def forward(self, nir_p, vis_g, vis_p, nir_g, cur_ids):
        nir_p = F.normalize(nir_p)
        vis_g = F.normalize(vis_g)
        vis_p = F.normalize(vis_p)
        nir_g = F.normalize(nir_g)
        batch_size = nir_p.shape[0]
        label = (torch.LongTensor([range(batch_size)]) + self.index)
        label = label.squeeze().cuda()
        nir_g = nir_g.detach()
        vis_g = vis_g.detach()
        output1 = self.compute_theta(nir_p, vis_g, label, batch_size)
        output2 = self.compute_theta(vis_p, nir_g, label, batch_size, False)
        output1 *= self.scale
        output2 *= self.scale
        self.update_queue(vis_g, nir_g, cur_ids, batch_size)
        id_set = self.get_id_set()
        return output1, output2, label, id_set
