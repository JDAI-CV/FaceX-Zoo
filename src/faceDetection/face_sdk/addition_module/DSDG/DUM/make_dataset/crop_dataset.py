import math
import os
import cv2
import numpy as np


root_dir = '/export2/home/wht/oulu_images_crop/'

img_root = '/export2/home/wht/oulu_images/train_img_flod/'
map_root = '/export2/home/wht/oulu_images/train_depth_flod/'
bbox_root = '/export2/home/wht/oulu_images/train_bbox_flod/'


def crop_face_from_scene(image, face_name_full, scale):
    f = open(face_name_full, 'r')
    lines = f.readlines()
    lines = lines[0].split(' ')
    y1, x1, w, h = [int(ele) for ele in lines[:4]]
    f.close()
    y2 = y1 + w
    x2 = x1 + h

    y_mid = (y1 + y2) / 2.0
    x_mid = (x1 + x2) / 2.0
    h_img, w_img = image.shape[0], image.shape[1]
    # w_img,h_img=image.size
    w_scale = scale * w
    h_scale = scale * h
    y1 = y_mid - w_scale / 2.0
    x1 = x_mid - h_scale / 2.0
    y2 = y_mid + w_scale / 2.0
    x2 = x_mid + h_scale / 2.0
    y1 = max(math.floor(y1), 0)
    x1 = max(math.floor(x1), 0)
    y2 = min(math.floor(y2), w_img)
    x2 = min(math.floor(x2), h_img)

    # region=image[y1:y2,x1:x2]
    region = image[x1:x2, y1:y2]
    return region


def crop_face_from_scene_prnet(image, face_name_full, scale):
    h_img, w_img = image.shape[0], image.shape[1]
    f = open(face_name_full, 'r')
    lines = f.readlines()
    lines = lines[0].split(' ')
    l, r, t, b = [int(ele) for ele in lines[:4]]
    if l < 0:
        l = 0
    if r > w_img:
        r = w_img
    if t < 0:
        t = 0
    if b > h_img:
        b = h_img
    y1 = l
    x1 = t
    w = r - l
    h = b - t
    f.close()
    y2 = y1 + w
    x2 = x1 + h

    y_mid = (y1 + y2) / 2.0
    x_mid = (x1 + x2) / 2.0
    # w_img,h_img=image.size
    w_scale = scale * w
    h_scale = scale * h
    y1 = y_mid - w_scale / 2.0
    x1 = x_mid - h_scale / 2.0
    y2 = y_mid + w_scale / 2.0
    x2 = x_mid + h_scale / 2.0
    y1 = max(math.floor(y1), 0)
    x1 = max(math.floor(x1), 0)
    y2 = min(math.floor(y2), w_img)
    x2 = min(math.floor(x2), h_img)

    # region=image[y1:y2,x1:x2]
    region = image[x1:x2, y1:y2]
    return region


vedio_list = os.listdir(bbox_root)
for i, vedio_name in enumerate(vedio_list):
    print(i)
    bbox_list = os.listdir(os.path.join(bbox_root, vedio_name))
    for bbox_name in bbox_list:

        face_scale = np.random.randint(12, 15)
        face_scale = face_scale / 10.0

        # face_scale = 1.3

        bbox_path = os.path.join(bbox_root, vedio_name, bbox_name)

        img_path = os.path.join(img_root, vedio_name, bbox_name[:-4] + '.jpg')

        img = cv2.imread(img_path)
        img_crop = cv2.resize(crop_face_from_scene_prnet(img, bbox_path, face_scale), (256, 256))

        img_crop_path = os.path.join(root_dir, 'train_img_flod')

        if not os.path.exists(os.path.join(img_crop_path, vedio_name)):
            os.makedirs(os.path.join(img_crop_path, vedio_name))

        cv2.imwrite(os.path.join(img_crop_path, vedio_name, bbox_name[:-4] + '.jpg'), img_crop)

        map_path = os.path.join(map_root, vedio_name, bbox_name[:-9] + 'depth1D.jpg')

        map = cv2.imread(map_path, 0)
        map_crop = cv2.resize(crop_face_from_scene_prnet(map, bbox_path, face_scale), (32, 32))
        map_crop_path = os.path.join(root_dir, 'train_depth_flod')
        if not os.path.exists(os.path.join(map_crop_path, vedio_name)):
            os.makedirs(os.path.join(map_crop_path, vedio_name))
        cv2.imwrite(os.path.join(map_crop_path, vedio_name, bbox_name[:-9] + 'depth1D.jpg'), map_crop)
