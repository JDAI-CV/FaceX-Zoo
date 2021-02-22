"""
@author: Yinglu Liu, Jun Wang
@date: 20201012
@contact: jun21wangustc@gmail.com
"""

from face_masker import FaceMasker

if __name__ == '__main__':
    is_aug = False
    image_path = 'Data/test-data/test1.jpg'
    face_lms_file = 'Data/test-data/test1_landmark.txt'
    template_name = '7.png'
    masked_face_path = 'test1_mask1.jpg'
    face_lms_str = open(face_lms_file).readline().strip().split(' ')
    face_lms = [float(num) for num in face_lms_str]
    face_masker = FaceMasker(is_aug)
    face_masker.add_mask_one(image_path, face_lms, template_name, masked_face_path)
