from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

'''
LwM is originally an incremental learning method with 
classification/distillation/attention distillation losses.

Here, LwM is only defined as the Grad-CAM based attention distillation.
'''
class LwM(nn.Module):
	'''
	Learning without Memorizing
	https://arxiv.org/pdf/1811.08051.pdf
	'''
	def __init__(self):
		super(LwM, self).__init__()

	def forward(self, out_s, fm_s, out_t, fm_t, target):
		target_out_t = torch.gather(out_t, 1, target.view(-1, 1))
		grad_fm_t    = grad(outputs=target_out_t, inputs=fm_t,
							grad_outputs=torch.ones_like(target_out_t),
							create_graph=True, retain_graph=True, only_inputs=True)[0]
		weights_t = F.adaptive_avg_pool2d(grad_fm_t, 1)
		cam_t = torch.sum(torch.mul(weights_t, grad_fm_t), dim=1, keepdim=True)
		cam_t = F.relu(cam_t)
		cam_t = cam_t.view(cam_t.size(0), -1)
		norm_cam_t = F.normalize(cam_t, p=2, dim=1)

		target_out_s = torch.gather(out_s, 1, target.view(-1, 1))
		grad_fm_s    = grad(outputs=target_out_s, inputs=fm_s,
							grad_outputs=torch.ones_like(target_out_s),
							create_graph=True, retain_graph=True, only_inputs=True)[0]
		weights_s = F.adaptive_avg_pool2d(grad_fm_s, 1)
		cam_s = torch.sum(torch.mul(weights_s, grad_fm_s), dim=1, keepdim=True)
		cam_s = F.relu(cam_s)
		cam_s = cam_s.view(cam_s.size(0), -1)
		norm_cam_s = F.normalize(cam_s, p=2, dim=1)

		loss = F.l1_loss(norm_cam_s, norm_cam_t.detach())

		return loss