from __future__ import print_function, division
import os
import sys
import math
import torch
import cv2
from PIL import Image
from skimage import io
from skimage import transform as ski_transform
from scipy import ndimage
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

def _gaussian(
        size=3, sigma=0.25, amplitude=1, normalize=False, width=None,
        height=None, sigma_horz=None, sigma_vert=None, mean_horz=0.5,
        mean_vert=0.5):
    # handle some defaults
    if width is None:
        width = size
    if height is None:
        height = size
    if sigma_horz is None:
        sigma_horz = sigma
    if sigma_vert is None:
        sigma_vert = sigma
    center_x = mean_horz * width + 0.5
    center_y = mean_vert * height + 0.5
    gauss = np.empty((height, width), dtype=np.float32)
    # generate kernel
    for i in range(height):
        for j in range(width):
            gauss[i][j] = amplitude * math.exp(-(math.pow((j + 1 - center_x) / (
                sigma_horz * width), 2) / 2.0 + math.pow((i + 1 - center_y) / (sigma_vert * height), 2) / 2.0))
    if normalize:
        gauss = gauss / np.sum(gauss)
    return gauss

def draw_gaussian(image, point, sigma):
    # Check if the gaussian is inside
    ul = [np.floor(np.floor(point[0]) - 3 * sigma),
          np.floor(np.floor(point[1]) - 3 * sigma)]
    br = [np.floor(np.floor(point[0]) + 3 * sigma),
          np.floor(np.floor(point[1]) + 3 * sigma)]
    if (ul[0] > image.shape[1] or ul[1] >
            image.shape[0] or br[0] < 1 or br[1] < 1):
        return image
    size = 6 * sigma + 1
    g = _gaussian(size)
    g_x = [int(max(1, -ul[0])), int(min(br[0], image.shape[1])) -
           int(max(1, ul[0])) + int(max(1, -ul[0]))]
    g_y = [int(max(1, -ul[1])), int(min(br[1], image.shape[0])) -
           int(max(1, ul[1])) + int(max(1, -ul[1]))]
    img_x = [int(max(1, ul[0])), int(min(br[0], image.shape[1]))]
    img_y = [int(max(1, ul[1])), int(min(br[1], image.shape[0]))]
    assert (g_x[0] > 0 and g_y[1] > 0)
    correct = False
    while not correct:
        try:
            image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]
            ] = image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]] + g[g_y[0] - 1:g_y[1], g_x[0] - 1:g_x[1]]
            correct = True
        except:
            print('img_x: {}, img_y: {}, g_x:{}, g_y:{}, point:{}, g_shape:{}, ul:{}, br:{}'.format(img_x, img_y, g_x, g_y, point, g.shape, ul, br))
            ul = [np.floor(np.floor(point[0]) - 3 * sigma),
                np.floor(np.floor(point[1]) - 3 * sigma)]
            br = [np.floor(np.floor(point[0]) + 3 * sigma),
                np.floor(np.floor(point[1]) + 3 * sigma)]
            g_x = [int(max(1, -ul[0])), int(min(br[0], image.shape[1])) -
                int(max(1, ul[0])) + int(max(1, -ul[0]))]
            g_y = [int(max(1, -ul[1])), int(min(br[1], image.shape[0])) -
                int(max(1, ul[1])) + int(max(1, -ul[1]))]
            img_x = [int(max(1, ul[0])), int(min(br[0], image.shape[1]))]
            img_y = [int(max(1, ul[1])), int(min(br[1], image.shape[0]))]
            pass
    image[image > 1] = 1
    return image

def transform(point, center, scale, resolution, rotation=0, invert=False):
    _pt = np.ones(3)
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = 200.0 * scale
    t = np.eye(3)
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)

    if rotation != 0:
        rotation = -rotation
        r = np.eye(3)
        ang = rotation * math.pi / 180.0
        s = math.sin(ang)
        c = math.cos(ang)
        r[0][0] = c
        r[0][1] = -s
        r[1][0] = s
        r[1][1] = c

        t_ = np.eye(3)
        t_[0][2] = -resolution / 2.0
        t_[1][2] = -resolution / 2.0
        t_inv = torch.eye(3)
        t_inv[0][2] = resolution / 2.0
        t_inv[1][2] = resolution / 2.0
        t = reduce(np.matmul, [t_inv, r, t_, t])

    if invert:
        t = np.linalg.inv(t)
    new_point = (np.matmul(t, _pt))[0:2]

    return new_point.astype(int)

def cv_crop(image, landmarks, center, scale, resolution=256, center_shift=0):
    new_image = cv2.copyMakeBorder(image, center_shift,
                                   center_shift,
                                   center_shift,
                                   center_shift,
                                   cv2.BORDER_CONSTANT, value=[0,0,0])
    new_landmarks = landmarks.copy()
    if center_shift != 0:
        center[0] += center_shift
        center[1] += center_shift
        new_landmarks = new_landmarks + center_shift
    length = 200 * scale
    top = int(center[1] - length // 2)
    bottom = int(center[1] + length // 2)
    left = int(center[0] - length // 2)
    right = int(center[0] + length // 2)
    y_pad = abs(min(top, new_image.shape[0] - bottom, 0))
    x_pad = abs(min(left, new_image.shape[1] - right, 0))
    top, bottom, left, right = top + y_pad, bottom + y_pad, left + x_pad, right + x_pad
    new_image = cv2.copyMakeBorder(new_image, y_pad,
                                   y_pad,
                                   x_pad,
                                   x_pad,
                                   cv2.BORDER_CONSTANT, value=[0,0,0])
    new_image = new_image[top:bottom, left:right]
    new_image = cv2.resize(new_image, dsize=(int(resolution), int(resolution)),
                           interpolation=cv2.INTER_LINEAR)
    new_landmarks[:, 0] = (new_landmarks[:, 0] + x_pad - left) * resolution / length
    new_landmarks[:, 1] = (new_landmarks[:, 1] + y_pad - top) * resolution / length
    return new_image, new_landmarks

def cv_rotate(image, landmarks, heatmap, rot, scale, resolution=256):
    img_mat = cv2.getRotationMatrix2D((resolution//2, resolution//2), rot, scale)
    ones = np.ones(shape=(landmarks.shape[0], 1))
    stacked_landmarks = np.hstack([landmarks, ones])
    new_landmarks = img_mat.dot(stacked_landmarks.T).T
    if np.max(new_landmarks) > 255 or np.min(new_landmarks) < 0:
        return image, landmarks, heatmap
    else:
        new_image = cv2.warpAffine(image, img_mat, (resolution, resolution))
        if heatmap is not None:
            new_heatmap = np.zeros((heatmap.shape[0], 64, 64))
            for i in range(heatmap.shape[0]):
                if new_landmarks[i][0] > 0:
                    new_heatmap[i] = draw_gaussian(new_heatmap[i],
                                                   new_landmarks[i]/4.0+1, 1)
        return new_image, new_landmarks, new_heatmap

def show_landmarks(image, heatmap, gt_landmarks, gt_heatmap):
    """Show image with pred_landmarks"""
    pred_landmarks = []
    pred_landmarks, _ = get_preds_fromhm(torch.from_numpy(heatmap).unsqueeze(0))
    pred_landmarks = pred_landmarks.squeeze()*4

    # pred_landmarks2 = get_preds_fromhm2(heatmap)
    heatmap = np.max(gt_heatmap, axis=0)
    heatmap = heatmap / np.max(heatmap)
    # image = ski_transform.resize(image, (64, 64))*255
    image = image.astype(np.uint8)
    heatmap = np.max(gt_heatmap, axis=0)
    heatmap = ski_transform.resize(heatmap, (image.shape[0], image.shape[1]))
    heatmap *= 255
    heatmap = heatmap.astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    plt.imshow(image)
    plt.scatter(gt_landmarks[:, 0], gt_landmarks[:, 1], s=0.5, marker='.', c='g')
    plt.scatter(pred_landmarks[:, 0], pred_landmarks[:, 1], s=0.5, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

def fan_NME(pred_heatmaps, gt_landmarks, num_landmarks=68):
    '''
       Calculate total NME for a batch of data

       Args:
           pred_heatmaps: torch tensor of size [batch, points, height, width]
           gt_landmarks: torch tesnsor of size [batch, points, x, y]

       Returns:
           nme: sum of nme for this batch
    '''
    nme = 0
    pred_landmarks, _ = get_preds_fromhm(pred_heatmaps)
    pred_landmarks = pred_landmarks.numpy()
    gt_landmarks = gt_landmarks.numpy()
    for i in range(pred_landmarks.shape[0]):
        pred_landmark = pred_landmarks[i] * 4.0
        gt_landmark = gt_landmarks[i]

        if num_landmarks == 68:
            left_eye = np.average(gt_landmark[36:42], axis=0)
            right_eye = np.average(gt_landmark[42:48], axis=0)
            norm_factor = np.linalg.norm(left_eye - right_eye)
            # norm_factor = np.linalg.norm(gt_landmark[36]- gt_landmark[45])
        elif num_landmarks == 98:
            norm_factor = np.linalg.norm(gt_landmark[60]- gt_landmark[72])
        elif num_landmarks == 19:
            left, top = gt_landmark[-2, :]
            right, bottom = gt_landmark[-1, :]
            norm_factor = math.sqrt(abs(right - left)*abs(top-bottom))
            gt_landmark = gt_landmark[:-2, :]
        elif num_landmarks == 29:
            # norm_factor = np.linalg.norm(gt_landmark[8]- gt_landmark[9])
            norm_factor = np.linalg.norm(gt_landmark[16]- gt_landmark[17])
        nme += (np.sum(np.linalg.norm(pred_landmark - gt_landmark, axis=1)) / pred_landmark.shape[0]) / norm_factor
    return nme

def fan_NME_hm(pred_heatmaps, gt_heatmaps, num_landmarks=68):
    '''
       Calculate total NME for a batch of data

       Args:
           pred_heatmaps: torch tensor of size [batch, points, height, width]
           gt_landmarks: torch tesnsor of size [batch, points, x, y]

       Returns:
           nme: sum of nme for this batch
    '''
    nme = 0
    pred_landmarks, _ = get_index_fromhm(pred_heatmaps)
    pred_landmarks = pred_landmarks.numpy()
    gt_landmarks = gt_landmarks.numpy()
    for i in range(pred_landmarks.shape[0]):
        pred_landmark = pred_landmarks[i] * 4.0
        gt_landmark = gt_landmarks[i]
        if num_landmarks == 68:
            left_eye = np.average(gt_landmark[36:42], axis=0)
            right_eye = np.average(gt_landmark[42:48], axis=0)
            norm_factor = np.linalg.norm(left_eye - right_eye)
        else:
            norm_factor = np.linalg.norm(gt_landmark[60]- gt_landmark[72])
        nme += (np.sum(np.linalg.norm(pred_landmark - gt_landmark, axis=1)) / pred_landmark.shape[0]) / norm_factor
    return nme

def power_transform(img, power):
    img = np.array(img)
    img_new = np.power((img/255.0), power) * 255.0
    img_new = img_new.astype(np.uint8)
    img_new = Image.fromarray(img_new)
    return img_new

def get_preds_fromhm(hm, center=None, scale=None, rot=None):
    max, idx = torch.max(
        hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
    idx += 1
    preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
    preds[..., 0].apply_(lambda x: (x - 1) % hm.size(3) + 1)
    preds[..., 1].add_(-1).div_(hm.size(2)).floor_().add_(1)

    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = torch.FloatTensor(
                    [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                     hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                preds[i, j].add_(diff.sign_().mul_(.25))

    preds.add_(-0.5)

    preds_orig = torch.zeros(preds.size())
    if center is not None and scale is not None:
        for i in range(hm.size(0)):
            for j in range(hm.size(1)):
                preds_orig[i, j] = transform(
                    preds[i, j], center, scale, hm.size(2), rot, True)

    return preds, preds_orig

def get_index_fromhm(hm):
    max, idx = torch.max(
        hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
    preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
    preds[..., 0].remainder_(hm.size(3))
    preds[..., 1].div_(hm.size(2)).floor_()

    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]), int(preds[i, j, 1])
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = torch.FloatTensor(
                    [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                     hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                preds[i, j].add_(diff.sign_().mul_(.25))

    return preds

def shuffle_lr(parts, num_landmarks=68, pairs=None):
    if num_landmarks == 68:
        if pairs is None:
            pairs = [[0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10],
                    [7, 9], [17, 26], [18, 25], [19, 24], [20, 23], [21, 22], [36, 45],
                    [37, 44], [38, 43], [39, 42], [41, 46], [40, 47], [31, 35], [32, 34],
                    [50, 52], [49, 53], [48, 54], [61, 63], [60, 64], [67, 65], [59, 55], [58, 56]]
    elif num_landmarks == 98:
        if pairs is None:
            pairs = [[0, 32], [1,31], [2, 30], [3, 29], [4, 28], [5, 27], [6, 26], [7, 25], [8, 24], [9, 23], [10, 22], [11, 21], [12, 20], [13, 19], [14, 18], [15, 17], [33, 46], [34, 45], [35, 44], [36, 43], [37, 42], [38, 50], [39, 49], [40, 48], [41, 47], [60, 72], [61, 71], [62, 70], [63, 69], [64, 68], [65, 75], [66, 74], [67, 73], [96, 97], [55, 59], [56, 58], [76, 82], [77, 81], [78, 80], [88, 92], [89, 91], [95, 93], [87, 83], [86, 84]]
    elif num_landmarks == 19:
        if pairs is None:
            pairs = [[0, 5], [1, 4], [2, 3], [6, 11], [7, 10], [8, 9], [12, 14], [15, 17]]
    elif num_landmarks == 29:
        if pairs is None:
            pairs = [[0, 1], [4, 6], [5, 7], [2, 3], [8, 9], [12, 14], [16, 17], [13, 15], [10, 11], [18, 19], [22, 23]]
    for matched_p in pairs:
        idx1, idx2 = matched_p[0], matched_p[1]
        tmp = np.copy(parts[idx1])
        np.copyto(parts[idx1], parts[idx2])
        np.copyto(parts[idx2], tmp)
    return parts


def generate_weight_map(weight_map,heatmap):

    k_size = 3
    dilate = ndimage.grey_dilation(heatmap ,size=(k_size,k_size))
    weight_map[np.where(dilate>0.2)] = 1
    return weight_map

def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )

    # Get the RGB buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring (fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (w, h, 3)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll (buf, 3, axis=2)
    return buf
