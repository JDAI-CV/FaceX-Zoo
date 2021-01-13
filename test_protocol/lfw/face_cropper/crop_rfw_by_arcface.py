"""
@author: Jun Wang 
@date: 20201012 
@contact: jun21wangustc@gmail.com
"""

import os
import math
import multiprocessing
import cv2
from core.image_cropper.arcface_face_recognition.FaceRecImageCropper import FaceRecImageCropper

def crop_rfw(rfw_root, rfw_lms_file, target_folder):
    face_cropper = FaceRecImageCropper()
    rfw_lms_file_buf = open(rfw_lms_file)
    image_name2lms = {}
    line = rfw_lms_file_buf.readline().strip()
    while line:
        line_strs = line.split()
        image_path_str = line_strs[0].split('/')
        image_name = image_path_str[-2] + '/' + image_path_str[-1]
        lms_str = line_strs[2:]
        assert(len(lms_str) == 10)
        lms = [float(s) for s in lms_str]
        image_name2lms[image_name] = lms
        line = rfw_lms_file_buf.readline().strip()

    for root, dirs, files in os.walk(rfw_root):
        for cur_file in files:
            cur_file_path = os.path.join(root, cur_file)
            cur_image = cv2.imread(cur_file_path)
            file_path_split = cur_file_path.split('/')
            image_name = file_path_split[-2] + '/' + file_path_split[-1]
            assert(image_name in image_name2lms)
            face_lms = image_name2lms[image_name]
            cur_cropped_image = face_cropper.crop_image_by_mat(cur_image, face_lms)
            target_path = os.path.join(target_folder, image_name)
            target_dir = os.path.dirname(target_path)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            cv2.imwrite(target_path, cur_cropped_image)

if __name__ == '__main__':
    rfw_root = '/export/home/wangjun492/wj_armory/faceX-Zoo/dataset/face_evaluation/rfw/RFW/images/test/data'
    rfw_face_info_folder = '/export/home/wangjun492/wj_armory/faceX-Zoo/dataset/face_evaluation/rfw/RFW/images/test/txts'
    target_folder = '/export/home/wangjun492/wj_armory/faceX-Zoo/dataset/face_evaluation/rfw/RFW_crop'

    African_root = os.path.join(rfw_root, 'African')
    African_lms_file = os.path.join(rfw_face_info_folder, 'African/African_lmk.txt')
    African_target_folder = os.path.join(target_folder, 'African')
    crop_rfw(African_root, African_lms_file, African_target_folder )

    Asian_root = os.path.join(rfw_root, 'Asian')
    Asian_lms_file = os.path.join(rfw_face_info_folder, 'Asian/Asian_lmk.txt')
    Asian_target_folder = os.path.join(target_folder, 'Asian')
    crop_rfw(Asian_root, Asian_lms_file, Asian_target_folder)

    Caucasian_root = os.path.join(rfw_root, 'Caucasian')
    Caucasian_lms_file = os.path.join(rfw_face_info_folder, 'Caucasian/Caucasian_lmk.txt')
    Caucasian_target_folder = os.path.join(target_folder, 'Caucasian')
    crop_rfw(Caucasian_root, Caucasian_lms_file, Caucasian_target_folder)

    Indian_root = os.path.join(rfw_root, 'Indian')
    Indian_lms_file = os.path.join(rfw_face_info_folder, 'Indian/Indian_lmk.txt')
    Indian_target_folder = os.path.join(target_folder, 'Indian')
    crop_rfw(Indian_root, Indian_lms_file, Indian_target_folder)
