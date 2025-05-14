# DSDG

Official Pytorch code of paper [Dual Spoof Disentanglement Generation for Face Anti-spoofing with Depth Uncertainty Learning](https://arxiv.org/pdf/2112.00568.pdf) in IEEE Transactions on Circuits and Systems for Video Technology.

Face anti-spoofing (FAS) plays a vital role in preventing face recognition systems from presentation attacks. Existing face anti-spoofing datasets lack diversity due to the insufficient identity and insignificant variance, which limits the generalization ability of FAS model. In this paper, we propose Dual Spoof Disentanglement Generation (DSDG) framework to tackle this challenge by "anti-spoofing via generation". Depending on the interpretable factorized latent disentanglement in Variational Autoencoder (VAE), DSDG learns a joint distribution of the identity representation and the spoofing pattern representation in the latent space. Then, large-scale paired live and spoofing images can be generated from random noise to boost the diversity of the training set. However, some generated face images are partially distorted due to the inherent defect of VAE. Such noisy samples are hard to predict precise depth values, thus may obstruct the widely-used depth supervised optimization. To tackle this issue, we further introduce a lightweight Depth Uncertainty Module (DUM), which alleviates the adverse effects of noisy samples by depth uncertainty learning. DUM is developed without extra-dependency, thus can be flexibly integrated with any depth supervised network for face anti-spoofing. We evaluate the effectiveness of the proposed method on five popular benchmarks and achieve state-of-the-art results under both intra- and inter- test settings.

## Requirements
Our experiments are conducted under the following environments:
- Python == 3.8
- Pytorch == 1.6.0
- torchvision == 0.7.0

## Training
Before training, we need to extract frame images for some video data sets. Then, we use [MTCNN](https://github.com/ipazc/mtcnn) for face detection and [PRNet](https://github.com/YadiraF/PRNet) for face depth map prediction. We give an example of the OULU-NPU dataset:
* Configure the paths in [./data/extract_frame.py](https://github.com/JDAI-CV/FaceX-Zoo/blob/main/addition_module/DSDG/data/extract_frame.py) to extract frames from videos.
* Configure the paths in [./data/bbox.py](https://github.com/JDAI-CV/FaceX-Zoo/blob/main/addition_module/DSDG/data/bbox.py) to get the location of face in each frame with MTCNN.
* Utilize the PRNet to get the depth map of face in each frame.
* Save the processed data in `./oulu_images/ `.

#### DSDG
* Download the LightCNN-29 model from this [link](https://drive.google.com/file/d/1Jn6aXtQ84WY-7J3Tpr2_j6sX0ch9yucS/view) and put it to `./ip_checkpoint`.
* Run [./data/make_train_list.py](https://github.com/JDAI-CV/FaceX-Zoo/blob/main/addition_module/DSDG/data/make_train_list.py) to generate training list.
* Run [./train_generator.sh](https://github.com/JDAI-CV/FaceX-Zoo/blob/main/addition_module/DSDG/train_generator.sh) to train the generator.
* Run [./generated.py](https://github.com/JDAI-CV/FaceX-Zoo/blob/main/addition_module/DSDG/generated.py) to generate paired face anti-spoofing data.
* Save the processed data in `./fake_images/` and utilize the PRNet to get the depth map.

#### DUM
* Run the [./DUM/make_dataset/crop_dataset.py](https://github.com/JDAI-CV/FaceX-Zoo/blob/main/addition_module/DSDG/DUM/make_dataset/crop_dataset.py) to crop the original data. Save the cropped data in `./oulu_images_crop`.
* Move the generated data in `./fake_images/` to `./oulu_images_crop/` and upgrade the protocol.
* Run [./DUM/train.py](https://github.com/JDAI-CV/FaceX-Zoo/blob/main/addition_module/DSDG/DUM/train.py) to train the model with original data and generated data.

## Testing
We provide a CDCN model with DUM trained on OULU-NPU Protocol-1, and the following shows how to test it.
* The trained model is released in `./DUM/checkpoint/`.
* Run [./DUM/test.py](https://github.com/JDAI-CV/FaceX-Zoo/blob/main/addition_module/DSDG/DUM/test.py) to test on OULU-NPU Protocol-1.

## Citation
Please consider citing our paper in your publications if the project helps your research.

## Acknowledgements
This repo is based on the following projects, thank the authors a lot.
- [BradyFU/DVG](https://github.com/BradyFU/DVG)
- [ZitongYu/CDCN](https://github.com/ZitongYu/CDCN)
- [YadiraF/PRNet](https://github.com/YadiraF/PRNet)
- [ipazc/mtcnn](https://github.com/ipazc/mtcnn)