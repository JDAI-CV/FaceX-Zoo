from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients
'''
Modified by https://github.com/bhheo/BSS_distillation
'''

def reduce_sum(x, keepdim=True):
	for d in reversed(range(1, x.dim())):
		x = x.sum(d, keepdim=keepdim)
	return x


def l2_norm(x, keepdim=True):
	norm = reduce_sum(x*x, keepdim=keepdim)
	return norm.sqrt()


class BSS(nn.Module):
	'''
	Knowledge Distillation with Adversarial Samples Supporting Decision Boundary
	https://arxiv.org/pdf/1805.05532.pdf
	'''
	def __init__(self, T):
		super(BSS, self).__init__()
		self.T = T

	def forward(self, attacked_out_s, attacked_out_t):
		loss = F.kl_div(F.log_softmax(attacked_out_s/self.T, dim=1),
						F.softmax(attacked_out_t/self.T, dim=1),
						reduction='batchmean') #* self.T * self.T

		return loss


class BSSAttacker():
	def __init__(self, step_alpha, num_steps, eps=1e-4):
		self.step_alpha = step_alpha
		self.num_steps = num_steps
		self.eps = eps

	def attack(self, model, img, target, attack_class):
		img = img.detach().requires_grad_(True)

		step = 0
		while step < self.num_steps:
			zero_gradients(img)
			_, _, _, _, _, output = model(img)

			score = F.softmax(output, dim=1)
			score_target = score.gather(1, target.unsqueeze(1))
			score_attack_class = score.gather(1, attack_class.unsqueeze(1))

			loss = (score_attack_class - score_target).sum()
			loss.backward()

			step_alpha = self.step_alpha * (target == output.max(1)[1]).float()
			step_alpha = step_alpha.unsqueeze(1).unsqueeze(1).unsqueeze(1)
			if step_alpha.sum() == 0:
				break

			pert = (score_target - score_attack_class).unsqueeze(1).unsqueeze(1)
			norm_pert = step_alpha * (pert + self.eps) * img.grad / l2_norm(img.grad)

			step_adv = img + norm_pert
			step_adv = torch.clamp(step_adv, -2.5, 2.5)
			img.data = step_adv.data

			step += 1

		return img
