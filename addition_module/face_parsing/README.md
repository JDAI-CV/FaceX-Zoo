# 
This repo hosts the face_parsing implementation of the CVPR2022 paper "General Facial Representation Learning in a Visual-Linguistic Manner"

# Some Results by FaRL
![image](Data/images/face_parsing.jpg)

# Requirements
* python >= 3.7.1
* pytorch >= 1.9.1

# Pre-trained Model
[face_parsing.farl.lapa]https://github.com/FacePerceiver/facer/releases/download/models-v1/face_parsing.farl.lapa.main_ema_136500_jit191.pt
Please put the pre-trained model under FaceX-Zoo/face_sdk/models/face_parsing/face_parsing_1.0/
# Usage
```sh
cd ../../face_sdk
python api_usage/face_parsing.py
```s

# Reference  
This project is mainly inspired by [FaRL](https://github.com/FacePerceiver/FaRL).
