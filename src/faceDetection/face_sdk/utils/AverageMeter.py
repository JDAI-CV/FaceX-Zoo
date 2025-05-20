# based on:
# https://github.com/pytorch/examples/blob/master/imagenet/main.py

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        #self.name = name
        #self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    meter = AverageMeter()
