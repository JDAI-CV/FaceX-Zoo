"""
@author: Jun Wang  
@date: 20201012
@contact: jun21wangustc@gmail.com
"""

import os
import sys
import math
import multiprocessing

import cv2
sys.path.append('/home/wangjun492/FaceX-Zoo/face_sdk')
from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper

def crop_ijbc(ijbc_root, pts_score_file, target_folder):
    face_cropper = FaceRecImageCropper()
    pts_score_file_buf = open(pts_score_file)
    line = pts_score_file_buf.readline().strip()
    while line:
        line_strs = line.split(' ')
        image_name = line_strs[0]
        image_path = os.path.join(ijbc_root, image_name)
        target_path = os.path.join(target_folder, image_name)
        cur_image = cv2.imread(image_path)        
        image_lmk = line_strs[1:11]
        face_lms = [float(s) for s in image_lmk]
        cur_cropped_image = face_cropper.crop_image_by_mat(cur_image, face_lms)
        cv2.imwrite(target_path, cur_cropped_image)
        line = pts_score_file_buf.readline().strip()
        
if __name__ == '__main__':
    ijbc_root = '/home/wangjun492/notespace/IJB_release/IJBC/loose_crop'
    pts_score_file = '/home/wangjun492/notespace/IJB_release/IJBC/meta/ijbc_name_5pts_score.txt'    
    target_folder = '/home/wangjun492/notespace/IJB_release/IJBC/arcface_crop'

    crop_ijbc(ijbc_root, pts_score_file, target_folder)
