"""
@author: JiXuan Xu, Jun Wang
@date: 20201015
@contact: jun21wangustc@gmail.com 
"""
# all self defined exception is derived from BuzException
class BuzException(Exception):
    pass

class InputError(BuzException):
    def __init__(self):
        pass
    def __str__(self):
        return ("Input type error!")
    
###############################################
#all image related exception.
###############################################
class ImageException(BuzException):
    pass

class EmptyImageError(ImageException):
    def __init__(self):
        pass
    def __str__(self):
        return ("The input image is empty.")

class FalseImageSizeError(ImageException):
    def __init__(self):
        pass
    def __str__(self):
        return ("The input image size is false.")
    
class FaseChannelError(ImageException):
    def __init__(self, channel):
        self.channel = channel
    def __str__(self):
        return ("Input channel {} is invalid(only 2, 3, 4 channel is support.),".format(repr(self.channel)))
