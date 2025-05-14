import math

def ramp_up(epoch, alpha, lamda=1):
    if epoch > alpha:
        return lamda
    else:
        w = lamda * math.exp(-5*math.pow((1-epoch/alpha), 2))
        return w

def ramp_down(epoch, alpha, lamda=1):
    if epoch < alpha:
        return lamda
    else:
        w = lamda * math.exp(-1*math.pow((1-alpha/epoch), 2))
        return w