import torch
import torch.nn as nn
import torch.nn.functional as F


class FitNet(nn.Module):
	'''
	FitNets: Hints for Thin Deep Nets
	https://arxiv.org/pdf/1412.6550.pdf
	'''
	def __init__(self):
		super(FitNet, self).__init__()

	def forward(self, fm_s, fm_t):
		loss = F.mse_loss(fm_s, fm_t)
		return loss
