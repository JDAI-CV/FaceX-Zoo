from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


'''
DML with only two networks
'''
class DML(nn.Module):
	'''
	Deep Mutual Learning
	https://zpascal.net/cvpr2018/Zhang_Deep_Mutual_Learning_CVPR_2018_paper.pdf
	'''
	def __init__(self):
		super(DML, self).__init__()

	def forward(self, out1, out2):
		loss = F.kl_div(F.log_softmax(out1, dim=1),
						F.softmax(out2, dim=1),
						reduction='batchmean')

		return loss
