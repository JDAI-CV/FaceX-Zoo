import argparse
import onnx
import torch
import numpy as np
import sys
sys.path.append('../../../')
from backbone.backbone_def import BackboneFactory
from test_protocol.utils.model_loader import ModelLoader

if __name__ == '__main__':
    conf = argparse.ArgumentParser(description='extract features for megaface.')
    conf.add_argument("--backbone_type", type = str,
                      help = "Resnet, Mobilefacenets.")
    conf.add_argument("--backbone_conf_file", type = str,
                      help = "The path of backbone_conf.yaml.")
    conf.add_argument('--model_path', type = str, default = 'mv_epoch_8.pt',
                      help = 'The path of model')
    conf.add_argument('--output_path', type = str, default = 'mv_epoch_8.onnx',
                      help = 'The output path of onnx model')
    parser.add_argument('--opset', type=int, default=11, help='opset version.')
    args = conf.parse_args()
    
    backbone_factory = BackboneFactory(args.backbone_type, args.backbone_conf_file)
    model_loader = ModelLoader(backbone_factory)
    model = model_loader.load_model(args.model_path)
    model.eval()
    img = np.random.randint(0, 255, size=(112,112,3), dtype=np.uint8)
    img = img[:,:,::-1].astype(np.float32)
    img = (img.transpose((2, 0, 1)) - 127.5) * 0.0078125
    img = torch.from_numpy(img).unsqueeze(0).float()
    torch.onnx.export(
        model,
        img,
        args.output_path,
        keep_initializers_as_inputs=False,
        verbose=True,
        opset_version=args.opset)
    onnx_model = onnx.load(args.output_path)
    graph = onnx_model.graph
    graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'
    onnx.save(onnx_model, args.output_path)
