# DMUE

Official Pytorch code of paper [Dive into Ambiguity: Latent Distribution Mining and Pairwise Uncertainty Estimation for Facial Expression Recognition](https://openaccess.thecvf.com/content/CVPR2021/papers/She_Dive_Into_Ambiguity_Latent_Distribution_Mining_and_Pairwise_Uncertainty_Estimation_CVPR_2021_paper.pdf) in CVPR2021.

Due to the subjective annotation and the inherent interclass similarity of facial expressions, one of key challenges in Facial Expression Recognition (FER) is the annotation ambiguity. In this paper, we proposes a solution, named DMUE, to address the problem of annotation ambiguity from two perspectives: the latent **D**istribution **M**ining and the pairwise **U**ncertainty **E**stimation. For the former, an auxiliary multi-branch learning framework is introduced to better mine and describe the latent distribution in the label space. For the latter, the pairwise relationship of semantic feature between instances are fully exploited to estimate the ambiguity extent in the instance space. The proposed method is independent to the backbone architectures, and brings no extra burden for inference. The experiments are conducted on the popular real-world benchmarks and the synthetic noisy datasets. Either way, the proposed DMUE stably achieves leading performance.

![Overall_Framework](https://github.com/JDAI-CV/FaceX-Zoo/blob/main/addition_module/DMUE/images/framework.png)

## Requirements
Our experiments are conducted under the following environments:
- Python 3.7
- Pytorch == 1.5.1
- torchvision == 0.6.0
- (Optional) apex from [this link](https://github.com/NVIDIA/apex.git)

In the following, we take ResNet-18 as the backbone and AffectNet as the benchmark to show how to use this repo.

## Pre-training
For pre-training on MS-Celeb-1M, please refer to the [pretrain](https://github.com/JDAI-CV/FaceX-Zoo/tree/main/addition_module/DMUE/pretrain) folder.
- Download MS-Celeb-1M and its five facial landmarks to your path. We provide a counterpart of landmark file [msra_lmk.txt](https://drive.google.com/drive/folders/1FQ_SOQ3zP0LwtX3iF65Wn1aNSiRRcMP9).
- Configure the paths in [./preprocess/crop_msra.py](https://github.com/JDAI-CV/FaceX-Zoo/blob/main/addition_module/DMUE/preprocess/crop_msra.py) and run it to crop faces.
- Configure the `data_root` in [./pretrain/gen_train_file.py](https://github.com/JDAI-CV/FaceX-Zoo/blob/main/addition_module/DMUE/pretrain/gen_train_file.py) and run it to generate training list.
- Configure the paths in [./pretrain/train_res18.sh](https://github.com/JDAI-CV/FaceX-Zoo/blob/main/addition_module/DMUE/pretrain/train_res18.sh) and run it to pre-train ResNet-18 on MS-Celeb-1M.

We provide a pre-trained ResNet-18 model in this [link](https://drive.google.com/drive/folders/1DqL6WHGFctrisfWlxklYXCgr1fWcvAvA).


## Training
Before training on AffectNet, the first step is to detect and align the faces. The alignment of face expression data is slightly different from the one of pre-trained data.
- Download the landmark detection model from this [link](https://drive.google.com/drive/folders/1qWWI5qRqghfLhT5gZI5HFIBIpbyFyJl6) and put it to `./preprocess/face_alignmenet/ckpt/`.
- Configure the paths in [./preprocess/crop_align.py](https://github.com/JDAI-CV/FaceX-Zoo/blob/main/addition_module/DMUE/preprocess/crop_align.py) and run it to detect and align faces.
- Configure the paths in [./preprocess/make_lmdb.py](https://github.com/JDAI-CV/FaceX-Zoo/blob/main/addition_module/DMUE/preprocess/make_lmdb.py) and run it to pack the images to a lmdb file.

Our method is trained on single GPU. Multi-GPUs with DataParallel may encounter Exceptions, since there are some batch splitting operations in the forward pass. Accordingly, we provide DistributedDataParallel for Multi-GPUs with the help of [apex](https://github.com/NVIDIA/apex.git). Note that accuracy is slightly dropped when using Multi-GPUs.
- Configure the paths in [config.py](https://github.com/JDAI-CV/FaceX-Zoo/blob/main/addition_module/DMUE/config.py).
- Run [train.sh](https://github.com/JDAI-CV/FaceX-Zoo/blob/main/addition_module/DMUE/train.sh) for single GPU / Run [train_ddp.sh](https://github.com/JDAI-CV/FaceX-Zoo/blob/main/addition_module/DMUE/train_ddp.sh) for Multi-GPUs.


## Testing
We provide a trained ResNet-18 model on AffectNet, and the following shows how to test it.
- Download the trained model in this [link](https://drive.google.com/drive/folders/1p_vRIClF5ZXdDVzQC0oYnffspA5TjqnU), and put it to `./checkpoints/`.
- Run [convert_weights.py](https://github.com/JDAI-CV/FaceX-Zoo/blob/main/addition_module/DMUE/convert_weights.py) to convert the multi-branches weights to the target-branch weights.
- Run [inference.py](https://github.com/JDAI-CV/FaceX-Zoo/blob/main/addition_module/DMUE/inference.py) to test on `./images/test1.jpg`.


## License
DMUE is released under the Apache License 2.0. Please see the [LICENSE](https://github.com/JDAI-CV/FaceX-Zoo/blob/main/LICENSE) file for more information.


## Citation
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follows.
```
@inproceedings{She2021DMUE,
  title     =  {Dive into Ambiguity: Latent Distribution Mining and Pairwise Uncertainty 
  		Estimation for Facial Expression Recognition},
  author    =  {Jiahui She, Yibo Hu, Hailin Shi, Jun Wang, Qiu Shen and Tao Mei},
  booktitle =  {CVPR},
  year      =  {2021}
}
```

## Acknowledgements
This repo is based on the following projects, thank the authors a lot.
- [deepinsight/insightface](https://github.com/deepinsight/insightface)
- [1adrianb/face-alignment](https://github.com/1adrianb/face-alignment)
- [ufoym/imbalanced-dataset-sampler](https://github.com/ufoym/imbalanced-dataset-sampler)