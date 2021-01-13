"""
@author: JiXuan Xu, Jun Wang
@date: 20201015
@contact: jun21wangustc@gmail.com 
"""
from abc import ABCMeta, abstractmethod

class BaseImageCropper(metaclass=ABCMeta):
    """Base class for all model loader.
    All image alignment classes need to inherit this base class.
    """
    def __init__(self):
        pass

    @abstractmethod
    def crop_image_by_mat(self, image, landmarks):
        """Should be overridden by all subclasses.
        Used for online image cropping, input the original Mat, 
        and return the Mat obtained from the image cropping.
        """
        pass
