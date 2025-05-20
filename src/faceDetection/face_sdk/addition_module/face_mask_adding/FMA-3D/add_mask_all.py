"""
@author: Yinglu Liu, Mingcan Xiang, Jun Wang
@date: 20201013
@contact: jun21wangustc@gmail.com
"""

import os
from face_masker import FaceMasker

def get_lms_templateName(face_info_file, image_name2template_name_file, masked_face_root):
    """Generate to do task list.

    Args:
        face_info_file: The file which contains image_name and landmarks.
        image_name2template_name_file: a mapping file
        masked_face_root: Targe folder to save masked images.
  
    Returns:
        image_name2lms dict, image_name2template_name dict.
    """
    image_name2lms = {}    
    face_info_file_buf = open(face_info_file)
    line = face_info_file_buf.readline().strip()
    while line:
        line_strs = line.split(' ')
        image_name = line_strs[0]
        image_lms = line_strs[2].split(',')
        assert(len(image_lms) == 106 * 2)
        image_lms = [float(num) for num in image_lms]
        image_name2lms[image_name] = image_lms
        target_path = os.path.join(masked_face_root, image_name)
        target_dir = os.path.dirname(target_path)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        line = face_info_file_buf.readline().strip()
    image_name2template_name = {}
    template_name_file_buf = open(image_name2template_name_file)
    line = template_name_file_buf.readline().strip()
    while line:
        image_name = line.split(' ')[0]
        template_name = line.split(' ')[1]
        image_name2template_name[image_name] = template_name
        line = template_name_file_buf.readline().strip()
    print('Total images: %d.' % len(image_name2lms))
    return image_name2lms, image_name2template_name

if __name__ == '__main__':
    is_aug = False
    image_name2template_name_file = '/export2/wangjun492/face_database/facex-zoo/share_file/test_data/megaface/facescrub2template_name.txt'
    face_root = '/export2/wangjun492/face_database/public_data/general_data/test_data/megaface/facescrub_new'
    face_info_file = '/export2/wangjun492/face_database/facex-zoo/share_file/test_data/megaface/facescrub_face_info.txt'
    masked_face_root = 'facescrub_mask'
    image_name2lms, image_name2template_name = get_lms_templateName(face_info_file, image_name2template_name_file, masked_face_root)
    face_masker = FaceMasker(is_aug)
    face_masker.add_mask(face_root, image_name2lms, image_name2template_name, masked_face_root)
