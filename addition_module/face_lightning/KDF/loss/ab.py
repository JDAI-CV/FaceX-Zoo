from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


class AB(nn.Module):
	'''
	Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons
	https://arxiv.org/pdf/1811.03233.pdf
	'''
	def __init__(self, margin):
		super(AB, self).__init__()

		self.margin = margin

	def forward(self, fm_s, fm_t):
		# fm befor activation
		loss = ((fm_s + self.margin).pow(2) * ((fm_s > -self.margin) & (fm_t <= 0)).float() +
			    (fm_s - self.margin).pow(2) * ((fm_s <= self.margin) & (fm_t > 0)).float())
		loss = loss.mean()

		return loss