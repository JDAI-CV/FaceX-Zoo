# FMA-3D
A method for adding a mask on a non-masked face image. Given a real masked face image (a) and a non-masked face image (d), we synthesize a photo-realistic masked face image with the mask from (a) and the facial area from (d). 
![image](Data/images/FMA-3D.jpg)

# Some Results by FMA-3D
![image](Data/images/mask-sample.jpg)

# Requirements
* python >= 3.7.1
* pytorch >= 1.1.0

# Usage
* Extract the landmarks.
You can extract the 106 landmarks by our [face sdk](../../../face_sdk) or any other methods.
* Add a mask on a face image.
You can refer to [add_mask_one.py](add_mask_one.py) as an example.
```sh
python add_mask_one.py
```

# Speed Up
Some advice:
* Write the whole process by multi-processing.
* Write the function of render in [face_masker.py](face_masker.py) by c++.

# Reference  
This project is mainly inspired by [PRNet](https://github.com/YadiraF/PRNet).
