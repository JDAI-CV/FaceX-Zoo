from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


'''
Adopted from https://github.com/passalis/probabilistic_kt/blob/master/nn/pkt.py
'''
class PKTCosSim(nn.Module):
	'''
	Learning Deep Representations with Probabilistic Knowledge Transfer
	http://openaccess.thecvf.com/content_ECCV_2018/papers/Nikolaos_Passalis_Learning_Deep_Representations_ECCV_2018_paper.pdf
	'''
	def __init__(self):
		super(PKTCosSim, self).__init__()

	def forward(self, feat_s, feat_t, eps=1e-6):
		# Normalize each vector by its norm
		feat_s_norm = torch.sqrt(torch.sum(feat_s ** 2, dim=1, keepdim=True))
		feat_s = feat_s / (feat_s_norm + eps)
		feat_s[feat_s != feat_s] = 0

		feat_t_norm = torch.sqrt(torch.sum(feat_t ** 2, dim=1, keepdim=True))
		feat_t = feat_t / (feat_t_norm + eps)
		feat_t[feat_t != feat_t] = 0

		# Calculate the cosine similarity
		feat_s_cos_sim = torch.mm(feat_s, feat_s.transpose(0, 1))
		feat_t_cos_sim = torch.mm(feat_t, feat_t.transpose(0, 1))

		# Scale cosine similarity to [0,1]
		feat_s_cos_sim = (feat_s_cos_sim + 1.0) / 2.0
		feat_t_cos_sim = (feat_t_cos_sim + 1.0) / 2.0

		# Transform them into probabilities
		feat_s_cond_prob = feat_s_cos_sim / torch.sum(feat_s_cos_sim, dim=1, keepdim=True)
		feat_t_cond_prob = feat_t_cos_sim / torch.sum(feat_t_cos_sim, dim=1, keepdim=True)

		# Calculate the KL-divergence
		loss = torch.mean(feat_t_cond_prob * torch.log((feat_t_cond_prob + eps) / (feat_s_cond_prob + eps)))

		return loss

