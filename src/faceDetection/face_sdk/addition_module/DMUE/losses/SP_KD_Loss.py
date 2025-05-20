import torch
import torch.nn.functional as F
import numpy as np


class SP_KD_Loss(object):
    def __call__(self, G_matrixs, G_main_matrixs):
        # G_matrixs: List
        # G_main_matrixs: List
        G_err = [F.mse_loss(G_aux, G_main) for G_aux, G_main in zip(G_matrixs, G_main_matrixs)]
        G_err = sum(G_err) / len(G_main_matrixs)

        return G_err


if __name__ == '__main__':
    loss = SoftLoss()
    x = torch.randn([2, 8])
    y = torch.from_numpy(np.array([7,2]))
    soft_y = torch.randn([2,8])
    mask = torch.ones(2,8).scatter_(1, y.view(-1,1).long(),0)
    soft_y = soft_y * mask
    Lsoft = loss(x, y, soft_y)
