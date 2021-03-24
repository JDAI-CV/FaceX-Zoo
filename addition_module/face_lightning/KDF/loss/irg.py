from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


class IRG(nn.Module):
	'''
	Knowledge Distillation via Instance Relationship Graph
	http://openaccess.thecvf.com/content_CVPR_2019/papers/
	Liu_Knowledge_Distillation_via_Instance_Relationship_Graph_CVPR_2019_paper.pdf

	The official code is written by Caffe
	https://github.com/yufanLIU/IRG
	'''
	def __init__(self, w_irg_vert, w_irg_edge, w_irg_tran):
		super(IRG, self).__init__()

		self.w_irg_vert = w_irg_vert
		self.w_irg_edge = w_irg_edge
		self.w_irg_tran = w_irg_tran

	def forward(self, irg_s, irg_t):
		fm_s1, fm_s2, feat_s, out_s = irg_s
		fm_t1, fm_t2, feat_t, out_t = irg_t

		loss_irg_vert = F.mse_loss(out_s, out_t)

		irg_edge_feat_s = self.euclidean_dist_feat(feat_s, squared=True)
		irg_edge_feat_t = self.euclidean_dist_feat(feat_t, squared=True)
		irg_edge_fm_s1  = self.euclidean_dist_fm(fm_s1, squared=True)
		irg_edge_fm_t1  = self.euclidean_dist_fm(fm_t1, squared=True)
		irg_edge_fm_s2  = self.euclidean_dist_fm(fm_s2, squared=True)
		irg_edge_fm_t2  = self.euclidean_dist_fm(fm_t2, squared=True)
		loss_irg_edge = (F.mse_loss(irg_edge_feat_s, irg_edge_feat_t) +
						 F.mse_loss(irg_edge_fm_s1,  irg_edge_fm_t1 ) +
						 F.mse_loss(irg_edge_fm_s2,  irg_edge_fm_t2 )) / 3.0

		irg_tran_s = self.euclidean_dist_fms(fm_s1, fm_s2, squared=True)
		irg_tran_t = self.euclidean_dist_fms(fm_t1, fm_t2, squared=True)
		loss_irg_tran = F.mse_loss(irg_tran_s, irg_tran_t)

		# print(self.w_irg_vert * loss_irg_vert)
		# print(self.w_irg_edge * loss_irg_edge)
		# print(self.w_irg_tran * loss_irg_tran)
		# print()

		loss = (self.w_irg_vert * loss_irg_vert +
				self.w_irg_edge * loss_irg_edge +
				self.w_irg_tran * loss_irg_tran)

		return loss

	def euclidean_dist_fms(self, fm1, fm2, squared=False, eps=1e-12):
		'''
		Calculating the IRG Transformation, where fm1 precedes fm2 in the network.
		'''
		if fm1.size(2) > fm2.size(2):
			fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(2), fm2.size(3)))
		if fm1.size(1) < fm2.size(1):
			fm2 = (fm2[:,0::2,:,:] + fm2[:,1::2,:,:]) / 2.0

		fm1 = fm1.view(fm1.size(0), -1)
		fm2 = fm2.view(fm2.size(0), -1)
		fms_dist = torch.sum(torch.pow(fm1-fm2, 2), dim=-1).clamp(min=eps)

		if not squared:
			fms_dist = fms_dist.sqrt()

		fms_dist = fms_dist / fms_dist.max()

		return fms_dist

	def euclidean_dist_fm(self, fm, squared=False, eps=1e-12): 
		'''
		Calculating the IRG edge of feature map. 
		'''
		fm = fm.view(fm.size(0), -1)
		fm_square = fm.pow(2).sum(dim=1)
		fm_prod   = torch.mm(fm, fm.t())
		fm_dist   = (fm_square.unsqueeze(0) + fm_square.unsqueeze(1) - 2 * fm_prod).clamp(min=eps)

		if not squared:
			fm_dist = fm_dist.sqrt()

		fm_dist = fm_dist.clone()
		fm_dist[range(len(fm)), range(len(fm))] = 0
		fm_dist = fm_dist / fm_dist.max()

		return fm_dist

	def euclidean_dist_feat(self, feat, squared=False, eps=1e-12):
		'''
		Calculating the IRG edge of feat.
		'''
		feat_square = feat.pow(2).sum(dim=1)
		feat_prod   = torch.mm(feat, feat.t())
		feat_dist   = (feat_square.unsqueeze(0) + feat_square.unsqueeze(1) - 2 * feat_prod).clamp(min=eps)

		if not squared:
			feat_dist = feat_dist.sqrt()

		feat_dist = feat_dist.clone()
		feat_dist[range(len(feat)), range(len(feat))] = 0
		feat_dist = feat_dist / feat_dist.max()

		return feat_dist