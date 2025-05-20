# Face evaluation protocal
## Test Data Preparation
### LFW  
* Test images download: http://vis-www.cs.umass.edu/lfw/lfw.tgz
* Pairs file download: http://vis-www.cs.umass.edu/lfw/pairs.txt
* Face bounding box and landmarks: [lfw_face_info.txt](../data/files/lfw_face_info.txt)  
* Code for cropping face: [crop_lfw_by_arcface.py](lfw/face_cropper/crop_lfw_by_arcface.py)
### CPLFW  
* Test images & pairs file & landmarks: http://www.whdeng.cn/CPLFW/index.html?reload=true#download  
* Code for cropping face: [crop_cplfw_by_arcface.py](lfw/face_cropper/crop_cplfw_by_arcface.py)
### CALFW  
* Test images & pairs file & landmarks: http://www.whdeng.cn/CALFW/index.html?reload=true#download
* Code for cropping face: [crop_calfw_by_arcface.py](lfw/face_cropper/crop_calfw_by_arcface.py)
### RFW  
* Test images & pairs file & landmarks : http://www.whdeng.cn/RFW/testing.html
* Code for cropping face: [crop_rfw_by_arcface.py](lfw/face_cropper/crop_rfw_by_arcface.py)  
### AgeDB30
* Test images & pairs file & landmarks: https://ibug.doc.ic.ac.uk/resources/agedb/
* Code for cropping face: [crop_agedb_by_arcface.py](lfw/face_cropper/crop_agedb_by_arcface.py)  
### MegaFace
* Web site: http://megaface.cs.washington.edu/
* Face bounding box and landmarks of Facescrub: [facescrub_face_info.txt](../data/files/facescrub_face_info.txt)
* Face bounding box and landmarks of Megaface: megaface_face_info.txt([Google](https://drive.google.com/file/d/1EubsMbKxaRbBCS4i9EgojGvteyhUCKjS/view?usp=sharing),[Baidu](https://pan.baidu.com/s/1UUYHA02JA4nYxm67t95DoQ):r04t)
* Code for cropping Facescrub: [crop_facescrub_by_arcface.py](megaface/face_cropper/crop_facescrub_by_arcface.py)  
* Code for cropping Megaface: [crop_megaface_by_arcface.py](megaface/face_cropper/crop_megaface_by_arcface.py)  
### MegaFace-mask
* facescrub2template_name: [facescrub2template_name.txt](../data/files/facescrub2template_name.txt)
* Code for cropping Facescrub: [crop_facescrub_by_arcface.py](megaface/face_cropper/crop_facescrub_by_arcface.py)  
* Code for cropping Megaface: [crop_megaface_by_arcface.py](megaface/face_cropper/crop_megaface_by_arcface.py)  
* Code for cropping upper-half face: [crop_megaface_by_arcface.py](megaface/face_cropper/crop_eye.py)  

## Common configuration  
(1) [backbone_conf.yaml](backbone_conf.yaml): the same with the one in training mode.  
(2) [data_conf.yaml](data_conf.yaml)
* pairs_file_path: the path of the official released pairs file.  
* cropped_face_folder: the directory which contains the  cropped faces.  
* image_list_file_path: the path of the cropped face images, which is a path relative to cropped_face_folder.  
* facescrub_list: the path of 'facescrub_features_list.json' released by MegaFace.  
* megaceface_list: the path of 'megaface_features_list.json_1000000_1' released by MegaFace.  
* facescrub_noises_file: the path of 'facescrub_noises.txt' released by insightface.  
* megaface_noises_file: the path of 'megaface_noises.txt' released by insightface.  
* megaface-mask: if 1, test on MegaFace-Mask, and 0 otherwise.

## Evaluation on LFW protocal  
Note: currently support LFW, CPLFW, CALFT, RFW and AgeDB.  
(1) modify the config in [test_lfw.sh](test_lfw.sh), and detailed description about configuration can be found in [test_lfw.py](test_lfw.py).  
(2) sh [test_lfw.sh](test_lfw.sh)  

## Evaluation on Megaface protocal
(1) modify the config in [extract_feature.sh](extract_feature.sh), and detailed description about configuration can be found in [extract_feature.py](extract_feature.py).  
(2) sh extract_feature.sh  
(3) modify the config in [remove_noises.sh](remove_noises.sh), and detailed description about configuration can be found in [remove_noises.py](remove_noises.py).  
(3) sh remove_noises.sh  
(4) modify the config in [test_megaface.sh](test_megaface.sh), and detailed description about configuration can be found in [test_megaface.py](test_megaface.py).  
(5) sh test_megaface.sh  

## Evaluation on Megaface-Mask
(1) Add mask on face images of Facescurb by FMA-3D. You can directly run [add_mask_all.py](../addition_module/face_mask_adding/FMA-3D/) with the following setting:  
```python
is_aug = False  
image_name2template_name_file = '' #the path of facescrub2template_name.txt  
face_root = '' #the root directory for facescrub.  
face_info_file = '' #the path of facescrub_face_info.txt  
masked_face_root: '' #the target root to save the masked facescrub.
```
Download these two files firstly: [facescrub2template_name.txt](../data/files/facescrub2template_name.txt), [facescrub_face_info.txt](../data/files/facescrub_face_info.txt). 

(2) Crop face from the masked face by [crop_facescrub_by_arcface.py](megaface/face_cropper/crop_facescrub_by_arcface.py).  
(3) Edit the config in [data_conf.yaml](data_conf.yaml).  
```yaml
megaface-mask : 1
masked_cropped_face_folder: #the root folder of the cropped and masked facescrub.
masked_image_list_file: #the relative path list of the facescrub.
```
(4) Evaluation  
```shell
sh extract_feature.sh
sh remove_noises.sh
sh test_megaface.sh
```
Note:  
1)The last parameter of 'CommonTestDataset' indicates whether to crop the upper-half face (eye part). You should set it to True in [extract_feature.py](extract_feature.py)(line 38, line 48) if you want to evaluate a model trained by upper-half face. Meanwhile, the 'out_h' in [backbone_conf.yaml](backbone_conf.yaml) should be set to 4 and the 'megaface-mask' in [data_conf.yaml](data_conf.yaml) should be set to 1.  
2)In order to evaluate the accuracy of two ensembled models (by concatenating features), you should first concatenate the features of two models by [feat_concat.py](utils/feat_concat.py), and then set 'is_concat' in [test_megaface.sh](test_megaface.sh) to 1.  

## More tips
* Please make sure that the 'model_loader.load_model()' can load your model successfully. Otherwise you should implement your 'load_model()' method in [model_loader.py](utils/model_loader.py).  
