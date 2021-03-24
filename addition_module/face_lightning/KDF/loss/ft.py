from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


class FT(nn.Module):
	'''
	araphrasing Complex Network: Network Compression via Factor Transfer
	http://papers.nips.cc/paper/7541-paraphrasing-complex-network-network-compression-via-factor-transfer.pdf
	'''
	def __init__(self):
		super(FT, self).__init__()

	def forward(self, factor_s, factor_t):
		loss = F.l1_loss(self.normalize(factor_s), self.normalize(factor_t))

		return loss

	def normalize(self, factor):
		norm_factor = F.normalize(factor.view(factor.size(0),-1))

		return norm_factor
