"""
@author: Jun Wang
@date: 20201014
@contact: jun21wangustc@gmail.com
"""

import os
import cv2

def crop_facescrub(facescrub_root, facescrub_img_list, target_folder):
    facescrub_img_list_buf = open(facescrub_img_list)
    line = facescrub_img_list_buf.readline().strip()
    while line:
        image_path = os.path.join(facescrub_root, line)
        target_path = os.path.join(target_folder, line)
        target_dir = os.path.dirname(target_path)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        image = cv2.imread(image_path)
        image = image[:60, :]
        cv2.imwrite(target_path, image)
        line = facescrub_img_list_buf.readline().strip()
            
if __name__ == '__main__':
    facescrub_root = '/export2/wangjun492/face_database/public_data/final_data/test_data/megaface/facescrub_mask_crop_arcface'
    facescrub_img_list = '/export2/wangjun492/face_database/public_data/meta_data/test_data/megaface/mid_files/facescrub_img_list.txt'
    target_folder = '/export2/wangjun492/face_database/public_data/final_data/test_data/megaface/facescrub_eye_crop'

    crop_facescrub(facescrub_root, facescrub_img_list, target_folder)
