"""
@author: fengyu, wangjun
@date: 20220620
@contact: fengyu_cnyc@163.com
"""

import sys
sys.path.append('.')
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
import logging.config
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('api')

import yaml
import cv2
import numpy as np
import torch
from utils.show import show_bchw
from utils.draw import draw_bchw
from core.model_loader.face_parsing.FaceParsingModelLoader import FaceParsingModelLoader
from core.model_handler.face_parsing.FaceParsingModelHandler import FaceParsingModelHandler
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler

with open('config/model_conf.yaml') as f:
    model_conf = yaml.load(f,Loader=yaml.FullLoader)

if __name__ == '__main__':
    # common setting for all models, need not modify.
    model_path = 'models'

    # face detection model setting.
    scene = 'non-mask'
    model_category = 'face_detection'
    model_name =  model_conf[scene][model_category]
    logger.info('Start to load the face detection model...')
    try:
        faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
        model, cfg = faceDetModelLoader.load_model()
        faceDetModelHandler = FaceDetModelHandler(model, 'cuda:0', cfg)
    except Exception as e:
        logger.error('Falied to load face detection Model.')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Success!')

    # face landmark model setting.
    model_category = 'face_alignment'
    model_name =  model_conf[scene][model_category]
    logger.info('Start to load the face landmark model...')
    try:
        faceAlignModelLoader = FaceAlignModelLoader(model_path, model_category, model_name)
        model, cfg = faceAlignModelLoader.load_model()
        faceAlignModelHandler = FaceAlignModelHandler(model, 'cuda:0', cfg)
    except Exception as e:
        logger.error('Failed to load face landmark model.')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Success!')        
        
    # face parsing model setting.
    scene = 'non-mask'
    model_category = 'face_parsing'
    model_name =  model_conf[scene][model_category]
    logger.info('Start to load the face parsing model...')
    try:
        faceParsingModelLoader = FaceParsingModelLoader(model_path, model_category, model_name)
        model, cfg = faceParsingModelLoader.load_model()
        faceParsingModelHandler = FaceParsingModelHandler(model, 'cuda:0', cfg)
    except Exception as e:
        logger.error('Falied to load face parsing Model.')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Success!')



    # read image and get face features.
    image_path = 'api_usage/test_images/test1.jpg'
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    try:
        dets = faceDetModelHandler.inference_on_image(image)
        face_nums = dets.shape[0]
        with torch.no_grad():
            for i in range(face_nums):
                landmarks = faceAlignModelHandler.inference_on_image(image, dets[i])

                landmarks = torch.from_numpy(landmarks[[104,105,54,84,90]]).float()
                if i == 0:
                    landmarks_five = landmarks
                else:
                    landmarks_five = torch.stack([landmarks_five,landmarks], dim = 0)   

            print(landmarks_five.shape)
            faces = faceParsingModelHandler.inference_on_image(face_nums, image, landmarks_five)
            seg_logits = faces['seg']['logits']


            seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
            show_bchw(draw_bchw(image, faces))  
            
    except Exception as e:
        logger.error('Parsing failed!')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Success!')
