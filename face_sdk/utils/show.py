# based on:
# https://github.com/FacePerceiver/facer/blob/main/facer/show.py
from typing import Optional, Tuple
import torch
from PIL import Image
import matplotlib.pyplot as plt
import math 

def bchw2hwc(images: torch.Tensor, nrows: Optional[int] = None, border: int = 2,
             background_value: float = 0) -> torch.Tensor:
    """ make a grid image from an image batch.
    Args:
        images (torch.Tensor): input image batch.
        nrows: rows of grid.
        border: border size in pixel.
        background_value: color value of background.
    """
    assert images.ndim == 4  # n x c x h x w
    images = images.permute(0, 2, 3, 1)  # n x h x w x c
    n, h, w, c = images.shape
    if nrows is None:
        nrows = max(int(math.sqrt(n)), 1)
    ncols = (n + nrows - 1) // nrows
    result = torch.full([(h + border) * nrows - border,
                         (w + border) * ncols - border, c], background_value,
                        device=images.device,
                        dtype=images.dtype)

    for i, single_image in enumerate(images):
        row = i // ncols
        col = i % ncols
        yy = (h + border) * row
        xx = (w + border) * col
        result[yy:(yy+h), xx:(xx+w), :] = single_image
    return result



def show_hwc(image: torch.Tensor):
    if image.dtype != torch.uint8:
        image = image.to(torch.uint8)
    if image.size(2) == 1:
        image = image.repeat(1, 1, 3)
    pimage = Image.fromarray(image.cpu().numpy())
    pimage.save("test.jpg")
    plt.imshow(pimage)
    plt.show()


def show_bchw(image: torch.Tensor):
    show_hwc(bchw2hwc(image))


    
