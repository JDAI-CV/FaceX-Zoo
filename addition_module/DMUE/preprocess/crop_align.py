import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage import transform as trans

from mtcnn import MTCNN
from face_alignmenet.get_landmark import *


img_root  = '/path/to/your/AffectNet_root'
save_root = '/path/to/your/align_larger_256'

missed_img_txt = '/path/to/your/missed_img.txt'

lb_txt  = '/path/to/your/AffectNet_label_file.txt'
lb2_txt = '/path/to/your/lb2.txt'


mtcnn = MTCNN(keep_all=True)
lmd_t = np.load('template.npy')
model_ft = load_model()
unsupported_format = ['tif','TIF', 'bmp', 'BMP']


lines = open(lb_txt, 'r').readlines()
for i in tqdm(range(len(lines))):
    line_ = lines[i]

    img_path = line_.split(' ')[0].split('_')[1]
    label = line_.split(' ')[1]

    img_folder, img_name = img_path.split('/')[0], img_path.split('/')[1]
    old_img_path = os.path.join(img_root, img_folder, img_name)
    new_img_path = os.path.join(save_root, img_folder, img_name)
    if not os.path.exists(os.path.dirname(new_img_path)):
        os.makedirs(os.path.dirname(new_img_path))
    if img_name.split('.')[1] in unsupported_format:
        img_name = img_name.split('.')[0]+'.jpg'
        for _format in unsupported_format:
            line_ = line_.replace(_format, '.jpg')

    if not os.path.exists(old_img_path):
        with open(missed_img_txt, 'a+') as f:
            f.write(line_+'\n')
        continue

    # align face
    img = Image.open(old_img_path)
    img_ = np.array(img)
    boxes, probs = mtcnn.detect(img, landmarks=False)
    if boxes is not None:
        if boxes[0][0] < 0:
            boxes[0][0] = 0
        if boxes[0][1] < 0:
            boxes[0][1] = 0
        if boxes[0][2] > img_.shape[1]:
            boxes[0][2] = img_.shape[1] - 1
        if boxes[0][3] > img_.shape[0]:
            boxes[0][3] = img_.shape[0] - 1
    else:
        with open(missed_img_txt, 'a+') as f:
            f.write(line_)
        continue

    img_c = img_[int(boxes[0][1]):int(boxes[0][3]), int(boxes[0][0]):int(boxes[0][2]), :]
    img_c = img_c[:, :, :3]
    img_re = cv2.resize(img_c, (256, 256), interpolation=cv2.INTER_CUBIC)
    img_re = img_re[:,:,:3]

    lmd1 = get_lmd(model_ft, img_re)
    lmd1[:, 0] = lmd1[:, 0] * img_c.shape[1] / 256 + boxes[0][0]
    lmd1[:, 1] = lmd1[:, 1] * img_c.shape[0] / 256 + boxes[0][1]

    tform = trans.SimilarityTransform()
    res = tform.estimate(lmd1, lmd_t)
    dst = trans.warp(img_, tform.inverse, output_shape=(800, 800), cval=0.0)
    dst = uint8(dst * 255)
    dst_ = Image.fromarray(dst)

    boxes, probs = mtcnn.detect(dst_, landmarks=False)
    if boxes is not None:
        if boxes[0][0] < 0:
            boxes[0][0] = 0
        if boxes[0][1] < 0:
            boxes[0][1] = 0
        if boxes[0][2] > dst.shape[1]:
            boxes[0][2] = dst.shape[1] - 1
        if boxes[0][3] > dst.shape[0]:
            boxes[0][3] = dst.shape[0] - 1

        h, w = int(boxes[0][3])-int(boxes[0][1]), int(boxes[0][2])-int(boxes[0][0])
        h_, w_ = h*256/224, w*256/224
        pdh, pdw = int((h_-h)/2), int((w_-w)/2)
        xmin, xmax, ymin, ymax = int(boxes[0][1])-pdh, int(boxes[0][3])+pdh, int(boxes[0][0])-pdw, int(boxes[0][2])+pdw
        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin = 0
        if ymax > dst.shape[1]:
            ymax = dst.shape[1] - 1
        if xmax > dst.shape[0]:
            xmax = dst.shape[0] - 1

        dst_large = dst[xmin:xmax, ymin:ymax, :]
        if dst_large.shape[2] > 3:
            dst_large = dst_large[:,:,:3]
    else:
        with open(missed_img_txt, 'a+') as f:
            f.write(line_)
        continue

    dst_large = cv2.resize(dst_large, (256, 256), interpolation=cv2.INTER_CUBIC)[:, :, :3]
    cv2.imwrite(new_img_path, dst_large[:,:,::-1])
    with open(lb2_txt, 'a+') as f:
        f.write(line_)




