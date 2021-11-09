# Training mode  
Two training modes are included currently, i.e., conventional training and [semi-siamese training](https://arxiv.org/abs/2007.08398). Edit the configuration of each training mode by the following steps, and then you can train a face recognition model by the certain mode.  

## 1. Training data
We use [MS-Celeb-1M-v1c](http://trillionpairs.deepglint.com/data) for conventional training. To perform open-set evaluation, we try our best to remove the identities which may overlap between this dataset and all of the test sets, resulting in a training set which includes 72,778 identities and about 3.28M images. The final identity list can be found in [MS-Celeb-1M-v1c-r_id_list.txt](../data/files/MS-Celeb-1M-v1c-r_id_list.txt). The format of training list should be the same as [MS-Celeb-1M-v1c-r_train_list.txt](../data/files/MS-Celeb-1M-v1c-r_train_list.txt). The shallow training set MS-Celeb-1M-v1c-Shallow is formed by randomly selecting two images of an identity in MS-Celeb-1M-v1c, and the selected image list can be downloaded in [MS-Celeb-1M-v1c-r-shallow_train_list.txt](../data/files/MS-Celeb-1M-v1c-r-shallow_train_list.txt). The training set for masked face recognition(MS-Celeb-1M-v1c-Mask) includes the original face images of each identity in MS-Celeb1M-v1c, as well as the corresponding masked face image by [FMA-3D](../addition_module/face_mask_adding/FMA-3D). 

## 2. Train a face recognition model
### Step1: Prepare the training data
Align the face images to 112*112 according to [face_align.py](../face_sdk/api_usage/face_crop.py).
### Step2: Configure the backbone  
Edit the configuration in [backbone_conf.yaml](backbone_conf.yaml). Detailed description about the configuration can be found in [backbone_def.py](../backbone/backbone_def.py).  
### Step3: Configure the head  
Edit the configuration in [head_conf.yaml](head_conf.yaml). Detailed description about the configuration can be found in [head_def.py](../head/head_def.py).  
### Step4: Configure the training setting for each mode.  
Edit the configuration in train.sh. Detailed description about the configuration can be found in train.py.  
### Step5: Start training  
sh train.sh
### Other Tips
* In order to train a model using only the upper half of face (model2 in 3.4), you need to set the last parameter of 'ImageDataset' to True and modify the 'out_h' of the backbone to 4.  
* In order to train the masked face recognition model (model3 in 3.4), you just need to change the training set to MS-Celeb-1M-v1c-Mask, which includes 72,778 identities and about 3.28*2M images.  

## 3. Trained models and logs
The models and training logs mentioned in our technical report are listed as follows. You can click the link to download them. For Megaface, we report the accuracy of the last checkpoint, and for other benchmarks, we report the accuracy of the best checkpoint.
### 3.1 Experiments of SOTA backbones
| Backbone | LFW | CPLFW | CALFW | AgeDb | MegaFace | Params | Macs | Models&Logs |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [MobileFaceNet](https://arxiv.org/abs/1804.07573)   | 99.57 | 83.33 | 93.82 | 95.97 | 90.39 | 1.19M | 227.57M | [Google](https://drive.google.com/drive/folders/1v8G_y4JzoVaxXGlt3iLtd6TIk0GYwA2c?usp=sharing),[Baidu](https://pan.baidu.com/s/1RqBkIqd3zCdpUO50DHpOIw):bmpn |
| [Resnet50-ir](https://arxiv.org/abs/1512.03385)     | 99.78 | 88.20 | 95.47 | 97.77 | 96.67 | 43.57M | 6.31G | [Google](https://drive.google.com/drive/folders/1s1O5YcoFFy5godV1velyIwq_CcXDXUrz?usp=sharing),[Baidu](https://pan.baidu.com/s/1W7LAAQ9jtA9jojpsrjI1Fg):8ecq |
| [Resnet152-irse](https://arxiv.org/abs/1709.01507)  | 99.85 | 89.72 | 95.56 | 98.13 | 97.48 | 71.14M | 12.33G | [Google](https://drive.google.com/drive/folders/1FzXobevacaQ-Y1NAhMjTKZCP3gu4I3ni?usp=sharing),[Baidu](https://pan.baidu.com/s/10Fhgn9fjjtqPLXgrYTaPlA):2d0c |
| [HRNet](https://arxiv.org/abs/1908.07919)           | 99.80 | 88.89 | 95.48 | 97.82 | 97.32 | 70.63M | 4.35G | [Google](https://drive.google.com/drive/folders/1Cr26ScPdfrScE4FD_CW1xZhBtLuGM85O?usp=sharing),[Baidu](https://pan.baidu.com/s/1nv36Fub8QiQZK0iV5aXl5Q):t9eo |
| [EfficientNet-B0](https://arxiv.org/abs/1905.11946) | 99.55 | 84.72 | 94.37 | 96.63 | 91.38 | 33.44M | 77.83M | [Google](https://drive.google.com/drive/folders/1wR48k8h8mCryMw4NrfkBtocw_TGp2S1q?usp=sharing),[Baidu](https://pan.baidu.com/s/1ZdLiQ_vJJxYYw6scohW9tA):sgja |
| [TF-NAS-A](https://arxiv.org/abs/2008.05314)        | 99.75 | 85.90 | 94.87 | 97.23 | 94.42 | 39.59M | 534.41M | [Google](https://drive.google.com/drive/folders/1vR17gH6NQXGAGUdaqUJqte8PhflzTkG1?usp=sharing),[Baidu](https://pan.baidu.com/s/1lFUVndOSrk4SVCMGPXrxQg):kq2v
| [LightCNN-29](https://arxiv.org/pdf/1511.02683.pdf) | 99.57 | 82.60 | 93.87 | 95.78 | 89.32 | 11.60M | 2.84G | [Google](https://drive.google.com/drive/folders/1EQeqt1n3q9LU46W0WqgXgWaf4fmzERam?usp=sharing),[Baidu]():kq2v|
| [GhostNet](https://arxiv.org/pdf/1911.11907.pdf)        | 99.65 | 83.52 | 93.93 | 95.70 | 89.42 | 26.76M | 194.49M | [Google](https://drive.google.com/drive/folders/1DI5JpgHG4x0GQIiO0--CIzWKNGa-TxXC?usp=sharing),[Baidu](https://pan.baidu.com/s/1q-UZycjyjVfWI_6_AjlQyA):6dg1 |
| [Attention-56](https://arxiv.org/abs/1704.06904)    | 99.88 | 89.18 | 95.65 | 98.12 | 97.75 | 98.96M | 6.34G | [Google](https://drive.google.com/drive/folders/1oxQ7EVxrCZ57MYjqPVqwIn8W4PtJ5G9m?usp=sharing),[Baidu](https://pan.baidu.com/s/1xcWw0GI_SesSQNp_ZNDqZg):f93u |
| [Attention-92(MX)](https://arxiv.org/abs/1704.06904)    | 99.82 | 90.33 | 95.88 | 98.08 | 98.09 | 134.56M | 10.62G | [Google](https://drive.google.com/drive/folders/1h_meJetsaVUm-37Wqo-o3ed9lyWcS8-B?usp=sharing),[Baidu](https://pan.baidu.com/s/1Vp6g_bS_2uBJ2OkHNzAxeQ):3ura |
| [ResNeSt50](https://hangzhang.org/files/resnest.pdf)    | 99.80 | 89.98 | 95.55 | 97.98 | 97.08 | 76.79M | 5.55G | [Google](https://drive.google.com/file/d/1v9waQnoQnniv8GdXHpiEUm148IRbJ9-P/view?usp=sharing),[Baidu]():3ura |
| [ReXNet_1.0](https://arxiv.org/pdf/2007.00992.pdf)    | 99.65 | 84.68 | 94.58 | 96.70 | 93.17 | 15.20M | 429.64M | [Google](https://drive.google.com/drive/folders/1bybc4psUaGF-4ucXoW3aloLOayCDs25U?usp=sharing),[Baidu]():3ura |
| [RepVGG_A0](https://arxiv.org/pdf/2101.03697.pdf)    | 99.77 | 85.43 | 94.88 | 96.97 | 94.40 | 39.94M | 1.55G | [Google](https://drive.google.com/drive/folders/1p6zTJNqzSvNq0JeT0BC9iT60lljIXUZ6?usp=sharing),[Baidu](https://pan.baidu.com/s/1uUd6Uv2Jg8VjPtdHN-H-0g):gdsf |
| [RepVGG_B0](https://arxiv.org/pdf/2101.03697.pdf)    | 99.72 | 86.77 | 95.17 | 97.57 | 95.75 | 46.65M | 3.44G | [Google](https://drive.google.com/drive/folders/1ueiMzZ0SFtMoH1rECDKYcUc1mjbXMFYF?usp=sharing),[Baidu](https://pan.baidu.com/s/1cJ-O67cCTzOSriIOWBnagg):ip68 |
| [RepVGG_B1](https://arxiv.org/pdf/2101.03697.pdf)    | 99.82 | 87.55 | 95.50 | 97.78 | 96.74 | 106.75M | 13.21G | [Google](https://drive.google.com/drive/folders/1SskjaThUZjQTI_IQ4MPoASGgomKbtaF7?usp=sharing),[Baidu](https://pan.baidu.com/s/1OOdwPajSGM6Greandy-gow):b60b |
| [Swin-T](https://arxiv.org/pdf/2103.14030.pdf)    | 99.87 | 88.57 | 95.56 | 97.90 | 97.83 | 46.74M | 4.37G | [Google](https://drive.google.com/drive/folders/1gngcXFxVnmw01f9p7lN89I4bJEEU1uz3?usp=sharing),[Baidu](https://pan.baidu.com/s/1O6VIoLbQxPDxAPqc5ec7-A):17ww |
| [Swin-S](https://arxiv.org/pdf/2103.14030.pdf)    | 99.85 | 90.03 | 95.92 | 98.05 | 98.17 | 68.01M | 8.53G | [Google](https://drive.google.com/drive/folders/1v0tyEKZ7YZacB2vjL7Y40m6XUf-dhyrv?usp=sharing),[Baidu](https://pan.baidu.com/s/1mTdbdTgEMEPgHCipT6Mb_Q):hhre |
* MegaFace means MegaFace rank1 accuracy.  
* Params and Macs are computed by [THOP](https://github.com/Lyken17/pytorch-OpCounter).  
* MX means mixed precision training by [apex](https://github.com/nvidia/apex).

### 3.2 Experiments of SOTA heads
| Supervisory Head | LFW | CPLFW | CALFW | AgeDb | MegaFace_rank1 | Models&Logs |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [AM-Softmax](https://arxiv.org/abs/1801.05599)     | 99.58 | 83.63 | 93.93 | 95.85 | 88.92 | [Google](https://drive.google.com/drive/folders/1UgeMteQ9LwlEYkfB5sUcrLQi2J-Q61_-?usp=sharing),[Baidu](https://pan.baidu.com/s/17jS7sDvvMvoMyJqGXIxpkQ):pe3n |
| [AdaM-Softmax](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_AdaptiveFace_Adaptive_Margin_and_Sampling_for_Face_Recognition_CVPR_2019_paper.pdf)   | 99.58 | 83.85 | 93.50 | 96.02 | 89.40 | [Google](https://drive.google.com/drive/folders/1Vxd3eagK6I_Dn0vXBmY2AXb6FoPgwU2k?usp=sharing),[Baidu](https://pan.baidu.com/s/1m7e4-SlHe52nSmhHrYupXQ):rcrk |
| [AdaCos](https://arxiv.org/abs/1905.00292)         | 99.65 | 83.27 | 92.63 | 95.38 | 82.95 | [Google](https://drive.google.com/drive/folders/1OdDK5l_LVdr-lPp6Ylr_uuIzAOiR2Ds-?usp=sharing),[Baidu](https://pan.baidu.com/s/1I3pw-nBPGaYA1gcEOAgG2w):3sef |
| [ArcFace](https://arxiv.org/abs/1801.07698)        | 99.57 | 83.68 | 93.98 | 96.23 | 88.39 | [Google](https://drive.google.com/drive/folders/10uBtximw8c7js21btcq5uqbhBSvK_zYD?usp=sharing),[Baidu](https://pan.baidu.com/s/1GhY9z69jZyZ6EbElDU32xQ):aujd |
| [MV-Softmax](https://arxiv.org/abs/1912.00833)     | 99.57 | 83.33 | 93.82 | 95.97 | 90.39 | [Google](https://drive.google.com/drive/folders/1JV69j5AGakBG2uwzy_KeL351hj5tPv6v?usp=sharing),[Baidu](https://pan.baidu.com/s/1GZv5Jb03dPrT2D219rUPbA):fcpd |
| [CurricularFace](https://arxiv.org/abs/2004.00288) | 99.60 | 83.03 | 93.75 | 95.82 | 87.27 | [Google](https://drive.google.com/drive/folders/1WE6kXxk43tIgK4AEROH9l0sOagIbsjSj?usp=sharing),[Baidu](https://pan.baidu.com/s/1Dakz7ldswhrp6Ypg2c4R7w):iru3 |
| [CircleLoss](https://arxiv.org/abs/2002.10857)     | 99.57 | 83.42 | 94.00 | 95.73 | 88.75 | [Google](https://drive.google.com/drive/folders/1pGgugVRuEo0oKr3zy4C4_X1b2P6wpPI5?usp=sharing),[Baidu](https://pan.baidu.com/s/1wXOi6sgZV6NvJCHmBQzSvQ):mj00 |
| [NPCFace](https://arxiv.org/abs/2007.10172)        | 99.55 | 83.80 | 94.13 | 95.87 | 89.13 | [Google](https://drive.google.com/drive/folders/1pc6IyqyPY4VvNft_xcuFu3wHuiaqQZ0S?usp=sharing),[Baidu](https://pan.baidu.com/s/1d43HBsonKl8xx3xwI6iblA):2hih |
| [MagFace](https://arxiv.org/pdf/2103.06627.pdf)    | 99.53 | 84.32 | 94.03 | 95.82 | 89.85 | [Google](https://drive.google.com/drive/folders/1yJZvfYAE3wmoBXdAIHHAOUXvX0uNaDcM?usp=sharing),[Baidu]():2hih |

### 3.3 Shallow Face Learning
| Training Mode | LFW | CPLFW | CALFW | AgeDb | Models&Logs |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Convention Training | 91.77 | 61.56 | 76.52 | 73.90 | [Google](https://drive.google.com/drive/folders/11Et8c2RuD3k7yy_qB-QzONWa3k5wIAeM?usp=sharing),[Baidu](https://pan.baidu.com/s/1VxPXurfd-PkjStFiVP3FVw):j4ve |
| [Semi-siamese Training](https://arxiv.org/abs/2007.08398) | 99.38 | 82.53  | 91.78 | 93.60 | [Google](https://drive.google.com/drive/folders/1EEY2UIofD0llYafZA7Lp6OlKd4t32o6K?usp=sharing),[Baidu](https://pan.baidu.com/s/1-7r3y9FzTPX9Wx88nVfusQ):n630 |

### 3.4 Masked Face Recognition
| Model | Rank1 | Rank3 | Rank5 | Rank10 | Models&Logs | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| model1 | 27.03 | 34.90 | 38.45 | 43.22 | [Google](https://drive.google.com/drive/folders/1F7iKdHJ3D6pAoyhzIUtw04PFXzJaDJH9?usp=sharing),[Baidu](https://pan.baidu.com/s/1GVusbkb0P7R7YjVPW2Laug):vp7e | Trained by MS-Celeb-1M-v1c |
| model2 | 71.40 | 76.60 | 78.62 | 81.05 | [Google](https://drive.google.com/drive/folders/1vWUAHpi0esYQdKImWQciwqYoLJS0AY5d?usp=sharing),[Baidu](https://pan.baidu.com/s/1HE7QO09UuleG3Wc62S72eg):b7tk | Trained by the upper half face in MS-Celeb-1M-v1c |
| model3 | 78.45 | 83.20 | 84.89 | 86.92 | [Google](https://drive.google.com/drive/folders/15ljHMkMoX6k7zvFdG8jFnef041wTIZU-?usp=sharing),[Baidu](https://pan.baidu.com/s/1mvk-1x1AGnck2TXX02-jCA):pcio | Trained by MS-Celeb-1M-v1c-Mask |
| model4 | 79.20 | 83.67 | 85.28 | 87.24 | [Google](https://drive.google.com/drive/folders/1mZBps6X61OxiSuys0Q0Vw7vc9EnLp8OQ?usp=sharing),[Baidu](https://pan.baidu.com/s/1Mb_-Rc_vyiaEZi483NdJjw):d9ii | Concat the features of model2 and model3 |

## 4. Extend the training module
### 4.1 Add new backbone
* Define the network under the directory [backbone](../backbone).  
* Create the object in [backbone_def.py](../backbone/backbone_def.py)  
* Add the configuration in [backbone_conf.yaml](backbone_conf.yaml).  
### 4.2 Add new head
* Define the new head under the directory [head](../head).  
* Create the object in [head_def.py](../head/head_def.py)  
* Add the configuration in [head_conf.yaml](head_conf.yaml).  
### 4.3 Add new data sampler
* Add the new data sampler in [train_dataset.py](../data_processor/train_dataset.py)  
### 4.4 Add new training mode
* Create a new folder named by the new training mode in this directory.  
* Implement the training procedure in 'train.py'.  
* Add the configuration in 'train.sh'.  
