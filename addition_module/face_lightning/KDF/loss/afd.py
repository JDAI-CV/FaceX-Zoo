from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

'''
In the original paper, AFD is one of components of AFDS.
AFDS: Attention Feature Distillation and Selection
AFD:  Attention Feature Distillation
AFS:  Attention Feature Selection

We find the original implementation of attention is unstable, thus we replace it with a SE block.
'''
class AFD(nn.Module):
	'''
	Pay Attention to Features, Transfer Learn Faster CNNs
	https://openreview.net/pdf?id=ryxyCeHtPB
	'''
	def __init__(self, in_channels, att_f):
		super(AFD, self).__init__()
		mid_channels = int(in_channels * att_f)

		self.attention = nn.Sequential(*[
				nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=True),
				nn.ReLU(inplace=True),
				nn.Conv2d(mid_channels, in_channels, 1, 1, 0, bias=True)
			])

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
		
	def forward(self, fm_s, fm_t, eps=1e-6):
		fm_t_pooled = F.adaptive_avg_pool2d(fm_t, 1)
		rho = self.attention(fm_t_pooled)
		# rho = F.softmax(rho.squeeze(), dim=-1)
		rho = torch.sigmoid(rho.squeeze())
		rho = rho / torch.sum(rho, dim=1, keepdim=True)

		fm_s_norm = torch.norm(fm_s, dim=(2,3), keepdim=True)
		fm_s      = torch.div(fm_s, fm_s_norm+eps)
		fm_t_norm = torch.norm(fm_t, dim=(2,3), keepdim=True)
		fm_t      = torch.div(fm_t, fm_t_norm+eps)

		loss = rho * torch.pow(fm_s-fm_t, 2).mean(dim=(2,3))
		loss = loss.sum(1).mean(0)

		return loss

# class AFD(nn.Module):
# 	'''
# 	Pay Attention to Features, Transfer Learn Faster CNNs
# 	https://openreview.net/pdf?id=ryxyCeHtPB
# 	'''
# 	def __init__(self, chw):
# 		super(AFD, self).__init__()
# 		c, h, w = chw

# 		self.weight1 = nn.Parameter(math.sqrt(2.0) / math.sqrt(h*w) * torch.randn(h, h*w))
# 		self.bias1   = nn.Parameter(torch.zeros(h))

# 		self.weight2 = nn.Parameter(math.sqrt(2.0) / math.sqrt(h) * torch.randn(h))
# 		self.bias2   = nn.Parameter(torch.zeros(c))

# 	def forward(self, fm_s, fm_t, eps=1e-6):
# 		b, c, h, w = fm_t.size()

# 		fm_t_flatten = fm_t.view(fm_t.size(0), fm_t.size(1), -1)
# 		weight1 = torch.stack([self.weight1.t()]*b, dim=0)
# 		bias1   = self.bias1.unsqueeze(0).unsqueeze(1)
# 		rho     = F.relu(torch.bmm(fm_t_flatten, weight1) + bias1)
# 		weight2 = self.weight2.view(-1, 1)
# 		bias2   = self.bias2.unsqueeze(0)
# 		rho     = torch.mm(rho.view(-1, rho.size(2)), weight2).view(b,c) + bias2
# 		# rho     = F.softmax(rho, dim=-1)
# 		rho = torch.sigmoid(rho)
# 		rho = rho / torch.sum(rho, dim=1, keepdim=True)
# 		# print(rho)

# 		fm_s_norm = torch.norm(fm_s, dim=(2,3), keepdim=True)
# 		fm_s      = torch.div(fm_s, fm_s_norm+eps)
# 		fm_t_norm = torch.norm(fm_t, dim=(2,3), keepdim=True)
# 		fm_t      = torch.div(fm_t, fm_t_norm+eps)

# 		loss = rho * torch.pow(fm_s-fm_t, 2).mean(dim=(2,3))
# 		loss = loss.sum(1).mean(0)

# 		return loss