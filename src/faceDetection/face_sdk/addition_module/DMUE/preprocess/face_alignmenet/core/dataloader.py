import sys
import os
import random
import glob
import torch
from skimage import io
from skimage import transform as ski_transform
from skimage.color import rgb2gray
import scipy.io as sio
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import Lambda, Compose
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue
from utils.utils import cv_crop, cv_rotate, draw_gaussian, transform, power_transform, shuffle_lr, fig2data, generate_weight_map
from PIL import Image
import cv2
import copy
import math


class AddBoundary(object):
    def __init__(self, num_landmarks=68):
        self.num_landmarks = num_landmarks

    def __call__(self, sample):
        landmarks_64 = np.floor(sample['landmarks'] / 4.0)
        if self.num_landmarks == 68:
            boundaries = {}
            boundaries['cheek'] = landmarks_64[0:17]
            boundaries['left_eyebrow'] = landmarks_64[17:22]
            boundaries['right_eyebrow'] = landmarks_64[22:27]
            boundaries['uper_left_eyelid'] = landmarks_64[36:40]
            boundaries['lower_left_eyelid'] = np.array([landmarks_64[i] for i in [36, 41, 40, 39]])
            boundaries['upper_right_eyelid'] = landmarks_64[42:46]
            boundaries['lower_right_eyelid'] = np.array([landmarks_64[i] for i in [42, 47, 46, 45]])
            boundaries['noise'] = landmarks_64[27:31]
            boundaries['noise_bot'] = landmarks_64[31:36]
            boundaries['upper_outer_lip'] = landmarks_64[48:55]
            boundaries['upper_inner_lip'] = np.array([landmarks_64[i] for i in [60, 61, 62, 63, 64]])
            boundaries['lower_outer_lip'] = np.array([landmarks_64[i] for i in [48, 59, 58, 57, 56, 55, 54]])
            boundaries['lower_inner_lip'] = np.array([landmarks_64[i] for i in [60, 67, 66, 65, 64]])
        elif self.num_landmarks == 98:
            boundaries = {}
            boundaries['cheek'] = landmarks_64[0:33]
            boundaries['left_eyebrow'] = landmarks_64[33:38]
            boundaries['right_eyebrow'] = landmarks_64[42:47]
            boundaries['uper_left_eyelid'] = landmarks_64[60:65]
            boundaries['lower_left_eyelid'] = np.array([landmarks_64[i] for i in [60, 67, 66, 65, 64]])
            boundaries['upper_right_eyelid'] = landmarks_64[68:73]
            boundaries['lower_right_eyelid'] = np.array([landmarks_64[i] for i in [68, 75, 74, 73, 72]])
            boundaries['noise'] = landmarks_64[51:55]
            boundaries['noise_bot'] = landmarks_64[55:60]
            boundaries['upper_outer_lip'] = landmarks_64[76:83]
            boundaries['upper_inner_lip'] = np.array([landmarks_64[i] for i in [88, 89, 90, 91, 92]])
            boundaries['lower_outer_lip'] = np.array([landmarks_64[i] for i in [76, 87, 86, 85, 84, 83, 82]])
            boundaries['lower_inner_lip'] = np.array([landmarks_64[i] for i in [88, 95, 94, 93, 92]])
        elif self.num_landmarks == 19:
            boundaries = {}
            boundaries['left_eyebrow'] = landmarks_64[0:3]
            boundaries['right_eyebrow'] = landmarks_64[3:5]
            boundaries['left_eye'] = landmarks_64[6:9]
            boundaries['right_eye'] = landmarks_64[9:12]
            boundaries['noise'] = landmarks_64[12:15]

        elif self.num_landmarks == 29:
            boundaries = {}
            boundaries['upper_left_eyebrow'] = np.stack([
                landmarks_64[0],
                landmarks_64[4],
                landmarks_64[2]
            ], axis=0)
            boundaries['lower_left_eyebrow'] = np.stack([
                landmarks_64[0],
                landmarks_64[5],
                landmarks_64[2]
            ], axis=0)
            boundaries['upper_right_eyebrow'] = np.stack([
                landmarks_64[1],
                landmarks_64[6],
                landmarks_64[3]
            ], axis=0)
            boundaries['lower_right_eyebrow'] = np.stack([
                landmarks_64[1],
                landmarks_64[7],
                landmarks_64[3]
            ], axis=0)
            boundaries['upper_left_eye'] = np.stack([
                landmarks_64[8],
                landmarks_64[12],
                landmarks_64[10]
            ], axis=0)
            boundaries['lower_left_eye'] = np.stack([
                landmarks_64[8],
                landmarks_64[13],
                landmarks_64[10]
            ], axis=0)
            boundaries['upper_right_eye'] = np.stack([
                landmarks_64[9],
                landmarks_64[14],
                landmarks_64[11]
            ], axis=0)
            boundaries['lower_right_eye'] = np.stack([
                landmarks_64[9],
                landmarks_64[15],
                landmarks_64[11]
            ], axis=0)
            boundaries['noise'] = np.stack([
                landmarks_64[18],
                landmarks_64[21],
                landmarks_64[19]
            ], axis=0)
            boundaries['outer_upper_lip'] = np.stack([
                landmarks_64[22],
                landmarks_64[24],
                landmarks_64[23]
            ], axis=0)
            boundaries['inner_upper_lip'] = np.stack([
                landmarks_64[22],
                landmarks_64[25],
                landmarks_64[23]
            ], axis=0)
            boundaries['outer_lower_lip'] = np.stack([
                landmarks_64[22],
                landmarks_64[26],
                landmarks_64[23]
            ], axis=0)
            boundaries['inner_lower_lip'] = np.stack([
                landmarks_64[22],
                landmarks_64[27],
                landmarks_64[23]
            ], axis=0)
        functions = {}

        for key, points in boundaries.items():
            temp = points[0]
            new_points = points[0:1, :]
            for point in points[1:]:
                if point[0] == temp[0] and point[1] == temp[1]:
                    continue
                else:
                    new_points = np.concatenate((new_points, np.expand_dims(point, 0)), axis=0)
                    temp = point
            points = new_points
            if points.shape[0] == 1:
                points = np.concatenate((points, points+0.001), axis=0)
            k = min(4, points.shape[0])
            functions[key] = interpolate.splprep([points[:, 0], points[:, 1]], k=k-1,s=0)

        boundary_map = np.zeros((64, 64))

        fig = plt.figure(figsize=[64/96.0, 64/96.0], dpi=96)

        ax = fig.add_axes([0, 0, 1, 1])

        ax.axis('off')

        ax.imshow(boundary_map, interpolation='nearest', cmap='gray')
        #ax.scatter(landmarks[:, 0], landmarks[:, 1], s=1, marker=',', c='w')

        for key in functions.keys():
            xnew = np.arange(0, 1, 0.01)
            out = interpolate.splev(xnew, functions[key][0], der=0)
            plt.plot(out[0], out[1], ',', linewidth=1, color='w')

        img = fig2data(fig)

        plt.close()

        sigma = 1
        temp = 255-img[:,:,1]
        temp = cv2.distanceTransform(temp, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        temp = temp.astype(np.float32)
        temp = np.where(temp < 3*sigma, np.exp(-(temp*temp)/(2*sigma*sigma)), 0 )

        fig = plt.figure(figsize=[64/96.0, 64/96.0], dpi=96)

        ax = fig.add_axes([0, 0, 1, 1])

        ax.axis('off')
        ax.imshow(temp, cmap='gray')
        plt.close()

        boundary_map = fig2data(fig)

        sample['boundary'] = boundary_map[:, :, 0]

        return sample

class AddWeightMap(object):
    def __call__(self, sample):
        heatmap= sample['heatmap']
        boundary = sample['boundary']
        heatmap = np.concatenate((heatmap, np.expand_dims(boundary, axis=0)), 0)
        weight_map = np.zeros_like(heatmap)
        for i in range(heatmap.shape[0]):
            weight_map[i] = generate_weight_map(weight_map[i],
                                                heatmap[i])
        sample['weight_map'] = weight_map
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, heatmap, landmarks, boundary, weight_map= sample['image'], sample['heatmap'], sample['landmarks'], sample['boundary'], sample['weight_map']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
            image_small = np.expand_dims(image_small, axis=2)
        image = image.transpose((2, 0, 1))
        boundary = np.expand_dims(boundary, axis=2)
        boundary = boundary.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float().div(255.0),
                'heatmap': torch.from_numpy(heatmap).float(),
                'landmarks': torch.from_numpy(landmarks).float(),
                'boundary': torch.from_numpy(boundary).float().div(255.0),
                'weight_map': torch.from_numpy(weight_map).float()}

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, img_dir, landmarks_dir, num_landmarks=68, gray_scale=False,
                 detect_face=False, enhance=False, center_shift=0,
                 transform=None,):
        """
        Args:
            landmark_dir (string): Path to the mat file with landmarks saved.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_dir = img_dir
        self.landmarks_dir = landmarks_dir
        self.num_lanmdkars = num_landmarks
        self.transform = transform
        self.img_names = glob.glob(self.img_dir+'*.jpg') + \
                         glob.glob(self.img_dir+'*.png')
        self.gray_scale = gray_scale
        self.detect_face = detect_face
        self.enhance = enhance
        self.center_shift = center_shift
        if self.detect_face:
            self.face_detector = MTCNN(thresh=[0.5, 0.6, 0.7])
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        pil_image = Image.open(img_name)
        if pil_image.mode != "RGB":
            # if input is grayscale image, convert it to 3 channel image
            if self.enhance:
                pil_image = power_transform(pil_image, 0.5)
            temp_image = Image.new('RGB', pil_image.size)
            temp_image.paste(pil_image)
            pil_image = temp_image
        image = np.array(pil_image)
        if self.gray_scale:
            image = rgb2gray(image)
            image = np.expand_dims(image, axis=2)
            image = np.concatenate((image, image, image), axis=2)
            image = image * 255.0
            image = image.astype(np.uint8)
        if not self.detect_face:
            center = [450//2, 450//2+0]
            if self.center_shift != 0:
                center[0] += int(np.random.uniform(-self.center_shift,
                                               self.center_shift))
                center[1] += int(np.random.uniform(-self.center_shift,
                                               self.center_shift))
            scale = 1.8
        else:
            detected_faces = self.face_detector.detect_image(image)
            if len(detected_faces) > 0:
                box = detected_faces[0]
                left, top, right, bottom, _ = box
                center = [right - (right - left) / 2.0,
                        bottom - (bottom - top) / 2.0]
                center[1] = center[1] - (bottom - top) * 0.12
                scale = (right - left + bottom - top) / 195.0
            else:
                center = [450//2, 450//2+0]
                scale = 1.8
            if self.center_shift != 0:
                shift = self.center * self.center_shift / 450
                center[0] += int(np.random.uniform(-shift, shift))
                center[1] += int(np.random.uniform(-shift, shift))
        base_name = os.path.basename(img_name)
        landmarks_base_name = base_name[:-4] + '_pts.mat'
        landmarks_name = os.path.join(self.landmarks_dir, landmarks_base_name)
        if os.path.isfile(landmarks_name):
            mat_data = sio.loadmat(landmarks_name)
            landmarks = mat_data['pts_2d']
        elif os.path.isfile(landmarks_name[:-8] + '.pts.npy'):
            landmarks = np.load(landmarks_name[:-8] + '.pts.npy')
        else:
            landmarks = []
            heatmap = []

        if landmarks != []:
            new_image, new_landmarks = cv_crop(image, landmarks, center,
                                               scale, 256, self.center_shift)
            tries = 0
            while self.center_shift != 0 and tries < 5 and (np.max(new_landmarks) > 240 or np.min(new_landmarks) < 15):
                center = [450//2, 450//2+0]
                scale += 0.05
                center[0] += int(np.random.uniform(-self.center_shift,
                                            self.center_shift))
                center[1] += int(np.random.uniform(-self.center_shift,
                                            self.center_shift))

                new_image, new_landmarks = cv_crop(image, landmarks,
                                                    center, scale, 256,
                                                    self.center_shift)
                tries += 1
            if np.max(new_landmarks) > 250 or np.min(new_landmarks) < 5:
                center = [450//2, 450//2+0]
                scale = 2.25
                new_image, new_landmarks = cv_crop(image, landmarks,
                                                    center, scale, 256,
                                                    100)
            assert (np.min(new_landmarks) > 0 and np.max(new_landmarks) < 256), \
                "Landmarks out of boundary!"
            image = new_image
            landmarks = new_landmarks
            heatmap = np.zeros((self.num_lanmdkars, 64, 64))
            for i in range(self.num_lanmdkars):
                if landmarks[i][0] > 0:
                    heatmap[i] = draw_gaussian(heatmap[i], landmarks[i]/4.0+1, 1)
        sample = {'image': image, 'heatmap': heatmap, 'landmarks': landmarks}
        if self.transform:
            sample = self.transform(sample)

        return sample

def get_dataset(val_img_dir, val_landmarks_dir, batch_size,
                num_landmarks=68, rotation=0, scale=0,
                center_shift=0, random_flip=False,
                brightness=0, contrast=0, saturation=0,
                blur=False, noise=False, jpeg_effect=False,
                random_occlusion=False, gray_scale=False,
                detect_face=False, enhance=False):
    val_transforms = transforms.Compose([AddBoundary(num_landmarks),
                                         AddWeightMap(),
                                         ToTensor()])

    val_dataset = FaceLandmarksDataset(val_img_dir, val_landmarks_dir,
                                       num_landmarks=num_landmarks,
                                       gray_scale=gray_scale,
                                       detect_face=detect_face,
                                       enhance=enhance,
                                       transform=val_transforms)

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   num_workers=0)
    data_loaders = {'val': val_dataloader}
    dataset_sizes = {}
    dataset_sizes['val'] = len(val_dataset)
    return data_loaders, dataset_sizes



