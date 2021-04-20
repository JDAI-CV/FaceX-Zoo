import torch
import torch.nn as nn
import torch.nn.functional as F

'''
NST with Polynomial Kernel, where d=2 and c=0
It can be treated as matching the Gram matrix of two vectorized feature map.
'''
class NST(nn.Module):
	def __init__(self):
		super(NST, self).__init__()

	def forward(self, g_s, g_t):
		#return [self.nst_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]
                return self.nst_loss(g_s, g_t)

	def nst_loss(self, f_s, f_t):
		s_H, t_H = f_s.shape[2], f_t.shape[2]
		s_W, t_W = f_s.shape[3], f_t.shape[3]
		if s_H > t_H:
			f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_W))
		elif s_H < t_H:
			f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_W))
		else:
			pass

		f_s = f_s.view(f_s.size(0), f_s.size(1), -1)
		f_s = F.normalize(f_s, dim=2)

		f_t = f_t.view(f_t.size(0), f_t.size(1), -1)
		f_t = F.normalize(f_t, dim=2)

		gram_s = self.gram_matrix(f_s)
		gram_t = self.gram_matrix(f_t)

		loss = F.mse_loss(gram_s, gram_t)

		return loss

	def gram_matrix(self, fm):
		return torch.bmm(fm, fm.transpose(1,2))
