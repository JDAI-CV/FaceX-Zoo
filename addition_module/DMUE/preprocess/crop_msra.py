import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from mtcnn import MTCNN

def mtcnn_detect(mtcnn, image):
    boxes, probs = mtcnn.detect(image, landmarks=False)

    if boxes is not None:
        if boxes[0][0] < 0:
            boxes[0][0] = 0
        if boxes[0][1] < 0:
            boxes[0][1] = 0
        if boxes[0][2] > image.size[1]:
            boxes[0][2] = image.size[1] - 1
        if boxes[0][3] > image.size[0]:
            boxes[0][3] = image.size[0] - 1
        return boxes[0]
    else:
        return None
    
def large_crop(boxes, image):
    img = np.array(image)
    boxes = [int(boxes[x]) for x in range(4)]
    
    h, w = boxes[3]-boxes[1], boxes[2]-boxes[0]
    h_, w_ = h*256/224, w*256/224
    pdh, pdw = int((h_-h)/2), int((w_-w)/2)
    xmin, xmax, ymin, ymax = boxes[1]-pdh, boxes[3]+pdh, boxes[0]-pdw, boxes[2]+pdw
    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0
    if ymax > img.shape[1]:
        ymax = img.shape[1] - 1
    if xmax > img.shape[0]:
        xmax = img.shape[0] - 1

    return img[xmin:xmax, ymin:ymax, :]

def align_crop(lines, ori_root, dst_root):
    mtcnn = MTCNN(keep_all=True)
    
    for i in tqdm(range(len(lines))):
        line = lines[i].strip()
        line_strs = line.split()
        image_name = line_strs[0]
        dst_image_path = os.path.join(dst_root, image_name)
        if not os.path.exists(os.path.dirname(dst_image_path)):
            os.makedirs(os.path.dirname(dst_image_path))

        image_path = os.path.join(ori_root, image_name)
        image = Image.open(image_path)
        boxes = mtcnn_detect(mtcnn, image)
        if boxes is not None:
            image_crop = large_crop(boxes, image)
            cv2.imwrite(dst_image_path, image_crop[:,:,::-1])
                       
if __name__== '__main__':
    lines = open('/path/to/your/msra_lmk.txt', 'r').readlines()
    ori_root = '/path/to/your/msra'
    dst_root = '/path/to/your/cropped_msra'

    align_crop(lines, ori_root, dst_root)
