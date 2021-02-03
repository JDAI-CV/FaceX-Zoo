"""
@author: Yinglu Liu, Jun Wang
@modifier: Champagne Jin (643683905@qq.com)
@date: 20201012
@contact: jun21wangustc@gmail.com
"""

from face_masker import FaceMasker

if __name__ == '__main__':
    is_aug = True
    image_path = 'Data/test-data/test1.jpg'
    template_name = '0.png'
    masked_face_path = 'test1_mask1_several.jpg'

    face_lms_files = ['Data/test-data/test1_landmark_res0.txt', 'Data/test-data/test1_landmark_res1.txt']
    face_lmses = []
    for face_lms_file in face_lms_files:
        face_lms_str = open(face_lms_file).readline().strip().split(' ')
        face_lmses.append([float(num) for num in face_lms_str])

    face_masker = FaceMasker(is_aug)
    face_masker.add_mask_multi(image_path, face_lmses, template_name, masked_face_path)
