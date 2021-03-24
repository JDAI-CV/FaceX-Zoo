from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad


class Sobolev(nn.Module):
	'''
	Sobolev Training for Neural Networks
	https://arxiv.org/pdf/1706.04859.pdf

	Knowledge Transfer with Jacobian Matching
	http://de.arxiv.org/pdf/1803.00443
	'''
	def __init__(self):
		super(Sobolev, self).__init__()

	def forward(self, out_s, out_t, img, target):
		target_out_s = torch.gather(out_s, 1, target.view(-1, 1))
		grad_s       = grad(outputs=target_out_s, inputs=img,
							grad_outputs=torch.ones_like(target_out_s),
							create_graph=True, retain_graph=True, only_inputs=True)[0]
		norm_grad_s  = F.normalize(grad_s.view(grad_s.size(0), -1), p=2, dim=1)

		target_out_t = torch.gather(out_t, 1, target.view(-1, 1))
		grad_t       = grad(outputs=target_out_t, inputs=img,
							grad_outputs=torch.ones_like(target_out_t),
							create_graph=True, retain_graph=True, only_inputs=True)[0]
		norm_grad_t  = F.normalize(grad_t.view(grad_t.size(0), -1), p=2, dim=1)

		loss = F.mse_loss(norm_grad_s, norm_grad_t.detach())

		return loss
