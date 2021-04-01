"""
@author: Jun Wang
@date: 20201012
@contact: jun21wangustc@gmail.com
"""

import os
import math
import multiprocessing
import cv2
import sys
sys.path.append('/export/home/wangjun492/wj_armory/faceX-Zoo/face_sdk')
from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper
from utils.lms_trans import lms106_2_lms25

def crop_facescrub(facescrub_root, facescrub_lms_file, target_folder):
    face_cropper = FaceRecImageCropper()
    facescrub_lms_file_buf = open(facescrub_lms_file)
    line = facescrub_lms_file_buf.readline().strip()
    while line:
        line_strs = line.split(' ')
        image_name = line_strs[0]
        bbox = line_strs[1]
        lms106 = line_strs[2]
        lms106 = lms106.split(',')
        assert(len(lms106) == 106 * 2)
        lms106 = [float(num) for num in lms106]
        cur_image_path = os.path.join(facescrub_root, image_name)
        assert(os.path.exists(cur_image_path))
        cur_image = cv2.imread(cur_image_path)
        cur_cropped_image = face_cropper.crop_image_by_mat(cur_image, lms106)
        target_path = os.path.join(target_folder, image_name)
        target_dir = os.path.dirname(target_path)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        cv2.imwrite(target_path, cur_cropped_image)
        line = facescrub_lms_file_buf.readline().strip()

if __name__ == '__main__':
    facescrub_root = '/export2/wangjun492/face_database/facex-zoo/private_file/test_data/megaface/facescrub_mask'
    facescrub_lms_file = '/export2/wangjun492/face_database/facex-zoo/share_file/test_data/megaface/facescrub_face_info.txt'
    target_folder = '/export2/wangjun492/face_database/facex-zoo/private_file/test_data/megaface/face_crop_arcface/masked_facescrub_crop'

    crop_facescrub(facescrub_root, facescrub_lms_file, target_folder)
