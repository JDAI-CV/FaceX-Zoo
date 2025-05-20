import torch
from imageCapture.camera import CameraHandler
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
torch.backends.cudnn.enabled = False

if __name__ == "__main__":
    camera = CameraHandler()
    camera.start()
