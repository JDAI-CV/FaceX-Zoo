"""
@author: Jun Wang
@date: 20210709
@contact: jun21wangustc@gmail.com
"""
import sys
import cv2
import onnx
import torch
import argparse
import onnxruntime
import numpy as np
sys.path.append('../../../')
from backbone.MobileFaceNets import MobileFaceNet

class OnnxConvertor:
    """Convert a pytorch model to a onnx model.
    """
    def __init__(self):
        pass
    def convert(self, model, output_path):
        img = np.random.randint(0, 255, size=(112,112,3), dtype=np.uint8)
        img = img[:,:,::-1].astype(np.float32)
        img = (img.transpose((2, 0, 1)) - 127.5) * 0.0078125
        img = torch.from_numpy(img).unsqueeze(0).float()
        torch.onnx.export(
            model,
            img,
            output_path,
            keep_initializers_as_inputs=False,
            verbose=True,
            opset_version=args.opset
        )
        # batch inference.
        onnx_model = onnx.load(output_path)
        graph = onnx_model.graph
        graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'
        onnx.save(onnx_model, output_path)

    def extract_feature_pytorch(self, model, image_path):
        image = cv2.imread(image_path)
        image = (image.transpose((2, 0, 1)) - 127.5) / 127.5
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = torch.unsqueeze(image, 0)
        with torch.no_grad():
            feature = model(image).cpu().numpy()
        feature = np.squeeze(feature)
        feature = feature/np.linalg.norm(feature)
        return feature

    def extract_feature_onnx(self, output_path, image_path):
        session =  onnxruntime.InferenceSession(output_path, None)
        input_cfg = session.get_inputs()[0]
        input_name = input_cfg.name
        outputs = session.get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)
        image = cv2.imread(image_path)
        images = []
        images.append(image)
        blob = cv2.dnn.blobFromImages(images, 1.0/127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=False)
        net_out = session.run(output_names, {input_name : blob})[0]
        feature = net_out[0]
        feature = feature/np.linalg.norm(feature)
        return feature
        
if __name__ == '__main__':
    conf = argparse.ArgumentParser(description='extract features for megaface.')
    conf.add_argument('--model_path', type = str, default = 'mv_epoch_8.pt',
                      help = 'The path of model')
    conf.add_argument('--output_path', type = str, default = 'mv_epoch_8.onnx',
                      help = 'The output path of onnx model')
    conf.add_argument('--opset', type=int, default=11, help='opset version.')
    args = conf.parse_args()
    image1 = 'test_images/11-FaceId-0_align.jpg'
    image2 = 'test_images/12-FaceId-0_align.jpg'
    
    model = MobileFaceNet(512, 7, 7)
    model_dict = model.state_dict()
    pretrained_dict = torch.load(args.model_path)['state_dict']
    new_pretrained_dict = {}
    for k in model_dict:
        new_pretrained_dict[k] = pretrained_dict['feat_net.'+k] 
    model_dict.update(new_pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()

    convertor = OnnxConvertor()
    convertor.convert(model, args.output_path)
    feature1_py = convertor.extract_feature_pytorch(model, image1)
    feature2_py = convertor.extract_feature_pytorch(model, image2)
    feature1_onnx = convertor.extract_feature_onnx(args.output_path, image1)
    feature2_onnx = convertor.extract_feature_onnx(args.output_path, image2)
    
    # check result.
    print(np.dot(feature1_py, feature2_py))
    print(np.dot(feature1_onnx, feature2_onnx))
    print(np.dot(feature1_py, feature1_onnx))
    print(np.dot(feature2_py, feature2_onnx))
