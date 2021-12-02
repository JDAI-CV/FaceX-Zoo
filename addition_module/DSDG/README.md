#DSDG

Official Pytorch code of paper Dual Spoof Disentanglement Generation for Face Anti-spoofing with Depth Uncertainty Learning in IEEE Transactions on Circuits and Systems for Video Technology.

##Requirements
Our experiments are conducted under the following environments:
* Python == 3.8
* Pytorch == 1.6.0
* torchvision == 0.7.0

##Training
Before training, we need to extract frame images for some video data sets. Then, we use [MTCNN](https://github.com/ipazc/mtcnn) for face detection and [PRNet](https://github.com/YadiraF/PRNet) for face depth map prediction. We give an example of the OULU-NPU dataset:
* Configure the paths in [./data/extract_frame.py]() to extract frames from videos.
* Configure the paths in [./data/bbox.py]() to get the location of face in each frame with MTCNN.
* Utilize the PRNet to get the depth map of face in each frame.
* Save the processed data in `./oulu_images/ `.
####DSDG
* Download the LightCNN-29 model from this [link](https://drive.google.com/file/d/1Jn6aXtQ84WY-7J3Tpr2_j6sX0ch9yucS/view) and put it to `./ip_checkpoint`.
* Run [./data/make_train_list.py]() to generate training list.
* Run [./train_generator.sh]() to train the generator.
* Run [./generated.py]() to generate paired face anti-spoofing data.
* Save the processed data in `./fake_images/` and utilize the PRNet to get the depth map.
####DUM
* Run the [./DUM/make_dataset/crop_dataset.py]() to crop the original data. Save the cropped data in `./oulu_images_crop`.
* Move the generated data in `./fake_images/` to `./oulu_images_crop/` and upgrade the protocol.
* Run [./DUM/train.py]() to train the model with original data and generated data.
##Testing
We provide a CDCN model with DUM trained on OULU-NPU Protocol-1, and the following shows how to test it.
* The trained model is released in `./DUM/checkpoint/`.
* Run [./DUM/test.py]() to test on OULU-NPU Protocol-1.
## Citation
Please consider citing our paper in your publications if the project helps your research.
##Acknowledgements
This repo is based on the following projects, thank the authors a lot.
* [BradyFU/DVG](https://github.com/BradyFU/DVG)
* [ZitongYu/CDCN](https://github.com/ZitongYu/CDCN)
* [YadiraF/PRNet](https://github.com/YadiraF/PRNet)
* [ipazc/mtcnn](https://github.com/ipazc/mtcnn)