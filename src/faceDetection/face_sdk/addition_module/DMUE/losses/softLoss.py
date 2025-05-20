import torch
import torch.nn.functional as F
import numpy as np


class SoftLoss(object):
    def __call__(self, outputs_x, targets_x, softlabel_x, epoch=None):
        # output_x: tensor
        # softlabel_x: tensor , size like output_x
        probs_x = torch.softmax(outputs_x, dim=1).cuda()
        mask = torch.ones_like(probs_x).scatter_(1, targets_x.view(-1, 1).long(), 0).cuda()
        probs_x = probs_x * mask
        Lsoft = torch.mean((probs_x - softlabel_x)**2)
        return Lsoft
    
class SoftLoss_normal(object):
    def __call__(self, outputs_x, targets_x, softlabel_x, epoch=None):
        # output_x: tensor
        # softlabel_x: list, each element size like output_x
        # it is the normal KD loss, used only in debug 
        print('The normal KD loss, used only in debug ')
        probs_x = torch.softmax(outputs_x, dim=1).cuda()
        kd_loss = [torch.mean((probs_x - aux_label)**2) for aux_label in softlabel_x]
        kd_loss = sum(kd_loss) / len(softlabel_x)
        
        return kd_loss

if __name__ == '__main__':
    loss = SoftLoss()
    x = torch.randn([2, 8])
    y = torch.from_numpy(np.array([7,2]))
    soft_y = torch.randn([2,8])
    mask = torch.ones(2,8).scatter_(1, y.view(-1,1).long(),0)
    soft_y = soft_y * mask
    Lsoft = loss(x, y, soft_y)
    