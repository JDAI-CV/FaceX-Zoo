# based on:
# https://github.com/FacePerceiver/facer/blob/main/facer/draw.py
from typing import Dict, List
import torch
import colorsys
import random
import numpy as np
from skimage.draw import line_aa, circle_perimeter_aa


def _gen_random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


_static_label_colors = [
    np.array((1.0, 1.0, 1.0), np.float32),
    np.array((255, 250, 79), np.float32) / 255.0,  # face
    np.array([255, 125, 138], np.float32) / 255.0,  # lb
    np.array([213, 32, 29], np.float32) / 255.0,  # rb
    np.array([0, 144, 187], np.float32) / 255.0,  # le
    np.array([0, 196, 253], np.float32) / 255.0,  # re
    np.array([255, 129, 54], np.float32) / 255.0,  # nose
    np.array([88, 233, 135], np.float32) / 255.0,  # ulip
    np.array([0, 117, 27], np.float32) / 255.0,  # llip
    np.array([255, 76, 249], np.float32) / 255.0,  # imouth
    np.array((1.0, 0.0, 0.0), np.float32),  # hair
    np.array((255, 250, 100), np.float32) / 255.0,  # lr
    np.array((255, 250, 100), np.float32) / 255.0,  # rr
    np.array((250, 245, 50), np.float32) / 255.0,  # neck
    np.array((0.0, 1.0, 0.5), np.float32),  # cloth
    np.array((1.0, 0.0, 0.5), np.float32),
] + _gen_random_colors(256)

_names_in_static_label_colors = [
    'background', 'face', 'lb', 'rb', 'le', 're', 'nose',
    'ulip', 'llip', 'imouth', 'hair', 'lr', 'rr', 'neck',
    'cloth', 'eyeg', 'hat', 'earr'
]

def select_data(selection, data):
    if isinstance(data, dict):
        return {name: select_data(selection, val) for name, val in data.items()}
    elif isinstance(data, (list, tuple)):
        return [select_data(selection, val) for val in data]
    elif isinstance(data, torch.Tensor):
        return data[selection]
    return data

def _blend_labels(image, labels, label_names_dict=None,
                  default_alpha=0.6, color_offset=None):
    assert labels.ndim == 2
    bg_mask = labels == 0
    if label_names_dict is None:
        colors = _static_label_colors
    else:
        colors = [np.array((1.0, 1.0, 1.0), np.float32)]
        for i in range(1, labels.max() + 1):
            if isinstance(label_names_dict, dict) and i not in label_names_dict:
                bg_mask = np.logical_or(bg_mask, labels == i)
                colors.append(np.zeros((3)))
                continue
            label_name = label_names_dict[i]
            if label_name in _names_in_static_label_colors:
                color = _static_label_colors[
                    _names_in_static_label_colors.index(
                        label_name)]
            else:
                color = np.array((1.0, 1.0, 1.0), np.float32)
            colors.append(color)

    if color_offset is not None:
        ncolors = []
        for c in colors:
            nc = np.array(c)
            if (nc != np.zeros(3)).any():
                nc += color_offset
            ncolors.append(nc)
        colors = ncolors

    if image is None:
        image = orig_image = np.zeros(
            [labels.shape[0], labels.shape[1], 3], np.float32)
        alpha = 1.0
    else:
        orig_image = image / np.max(image)
        image = orig_image * (1.0 - default_alpha)
        alpha = default_alpha
    for i in range(1, np.max(labels) + 1):
        image += alpha * \
            np.tile(
                np.expand_dims(
                    (labels == i).astype(np.float32), -1),
                [1, 1, 3]) * colors[(i) % len(colors)]
    image[np.where(image > 1.0)] = 1.0
    image[np.where(image < 0)] = 0.0
    image[np.where(bg_mask)] = orig_image[np.where(bg_mask)]
    return image


def _draw_hwc(image: torch.Tensor, data: Dict[str, torch.Tensor]):
    dtype = image.dtype
    h, w, _ = image.shape

    for tag, batch_content in data.items():
        if tag == 'points':
            for content in batch_content:
                # content: npoints x 2
                for x, y in content:
                    x = max(min(int(x), w-1), 0)
                    y = max(min(int(y), h-1), 0)
                    rr, cc, val = circle_perimeter_aa(y, x, 1)
                    valid = np.all([rr >= 0, rr < h, cc >= 0, cc < w], axis=0)
                    rr = rr[valid]
                    cc = cc[valid]
                    val = val[valid]
                    val = val[:, None][:, [0, 0, 0]]
                    image[rr, cc] = image[rr, cc] * (1.0-val) + val * 255

        if tag == 'seg':
            label_names = batch_content['label_names']
            for seg_logits in batch_content['logits']:
                # content: nclasses x h x w
                seg_probs = seg_logits.softmax(dim=0)
                seg_labels = seg_probs.argmax(dim=0).cpu().numpy()
                image = (_blend_labels(image.astype(np.float32) /
                         255, seg_labels,
                         label_names_dict=label_names) * 255).astype(dtype)

    return torch.from_numpy(image).cuda()


def draw_bchw(images, data):
    image = _draw_hwc(images, data).permute(2, 0, 1).unsqueeze(0)
    return image