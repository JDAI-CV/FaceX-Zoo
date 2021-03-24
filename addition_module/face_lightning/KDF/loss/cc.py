from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


'''
CC with P-order Taylor Expansion of Gaussian RBF kernel
'''
class CC(nn.Module):
	'''
	Correlation Congruence for Knowledge Distillation
	http://openaccess.thecvf.com/content_ICCV_2019/papers/
	Peng_Correlation_Congruence_for_Knowledge_Distillation_ICCV_2019_paper.pdf
	'''
	def __init__(self, gamma, P_order):
		super(CC, self).__init__()
		self.gamma = gamma
		self.P_order = P_order

	def forward(self, feat_s, feat_t):
		corr_mat_s = self.get_correlation_matrix(feat_s)
		corr_mat_t = self.get_correlation_matrix(feat_t)

		loss = F.mse_loss(corr_mat_s, corr_mat_t)

		return loss

	def get_correlation_matrix(self, feat):
		feat = F.normalize(feat, p=2, dim=-1)
		sim_mat  = torch.matmul(feat, feat.t())
		corr_mat = torch.zeros_like(sim_mat)

		for p in range(self.P_order+1):
			corr_mat += math.exp(-2*self.gamma) * (2*self.gamma)**p / \
						math.factorial(p) * torch.pow(sim_mat, p)

		return corr_mat
