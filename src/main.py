import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from imageCapture.camera import CameraHandler

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
torch.backends.cudnn.enabled = False

if __name__ == "__main__":
    camera = CameraHandler()
    camera.start()
