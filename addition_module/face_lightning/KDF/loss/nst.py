from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


# '''
# NST with Polynomial Kernel, where d=2 and c=0
# It can be treated as matching the Gram matrix of two vectorized feature map.
# '''
# class NST(nn.Module):
# 	def __init__(self):
# 		super(NST, self).__init__()

# 	def forward(self, fm_s, fm_t):
# 		fm_s = fm_s.view(fm_s.size(0), fm_s.size(1), -1)
# 		fm_s = F.normalize(fm_s, dim=2)

# 		fm_t = fm_t.view(fm_t.size(0), fm_t.size(1), -1)
# 		fm_t = F.normalize(fm_t, dim=2)

# 		gram_s = self.gram_matrix(fm_s)
# 		gram_t = self.gram_matrix(fm_t)

# 		loss = F.mse_loss(gram_s, gram_t)

# 		return loss

# 	def gram_matrix(self, fm):
# 		return torch.bmm(fm, fm.transpose(1,2))


'''
NST with Polynomial Kernel, where d=2 and c=0
'''
class NST(nn.Module):
	'''
	Like What You Like: Knowledge Distill via Neuron Selectivity Transfer
	https://arxiv.org/pdf/1707.01219.pdf
	'''
	def __init__(self):
		super(NST, self).__init__()

	def forward(self, fm_s, fm_t):
		fm_s = fm_s.view(fm_s.size(0), fm_s.size(1), -1)
		fm_s = F.normalize(fm_s, dim=2)

		fm_t = fm_t.view(fm_t.size(0), fm_t.size(1), -1)
		fm_t = F.normalize(fm_t, dim=2)

		loss = self.poly_kernel(fm_t, fm_t).mean() \
			 + self.poly_kernel(fm_s, fm_s).mean() \
			 - 2 * self.poly_kernel(fm_s, fm_t).mean()

		return loss

	def poly_kernel(self, fm1, fm2):
		fm1 = fm1.unsqueeze(1)
		fm2 = fm2.unsqueeze(2)
		out = (fm1 * fm2).sum(-1).pow(2)

		return out
