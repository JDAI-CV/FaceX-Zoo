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
from core.image_cropper.arcface_face_recognition.FaceRecImageCropper import FaceRecImageCropper
from utils.lms_trans import lms106_2_lms25

def crop_agedb(agedb_root, target_folder):
    face_cropper = FaceRecImageCropper()
    imageName2lms = {}
    for root, dirs, files in os.walk(agedb_root):
        for cur_file in files:
            if cur_file.endswith('.pts'):
                cur_image_name = cur_file.replace('.pts', '.jpg')
                cur_file_path = os.path.join(root, cur_file)
                cur_file_buf = open(cur_file_path)
                line = cur_file_buf.readline().strip()
                cur_lms = []
                while line:
                    line_strs = line.split(' ')
                    if len(line_strs) < 2:
                        line = cur_file_buf.readline().strip()
                        continue
                    if line_strs[0][0].isalpha():
                        line = cur_file_buf.readline().strip()
                        continue
                    cur_point = [float(num) for num in line_strs]
                    cur_lms.extend(cur_point)
                    line = cur_file_buf.readline().strip()
                assert(len(cur_lms) == 68 * 2)
                imageName2lms[cur_image_name] = cur_lms

    for image_name, lms in imageName2lms.items():
        image_path = os.path.join(agedb_root, image_name)
        target_path = os.path.join(target_folder, image_name)
        os.path.exists(image_path)
        cur_image = cv2.imread(image_path)
        cur_cropped_image = face_cropper.crop_image_by_mat(cur_image, lms)
        cv2.imwrite(target_path, cur_cropped_image)
                
if __name__ == '__main__':
    agedb_root = '/export/home/wangjun492/wj_armory/faceX-Zoo/dataset/face_evaluation/agedb/03_Protocol_Images'
    target_folder = '/export/home/wangjun492/wj_armory/faceX-Zoo/dataset/face_evaluation/agedb/03_Protocol_Images_crop'

    crop_agedb(agedb_root, target_folder)
