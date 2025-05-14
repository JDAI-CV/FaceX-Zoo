#!/bin/bash

echo train generator

gpu_ids='0,1,2,3'
workers=8
batch_size=240
all_epochs=200
pre_epoch=0
hdim=128
attack_type=4
test_epoch=10
save_epoch=10

lambda_mmd=50
lambda_ip=1000
lambda_pair=5
lambda_type=10
lambda_ort=1

ip_model='./ip_checkpoint/LightCNN_29Layers_V2_checkpoint.pth.tar'
img_root='/export2/home/wht/oulu_images/'
train_list='./train_list/train_list_oulu_p1.txt'
out_path='./results_oulu_p1/'
checkpoint_path='./model_oulu_p1/'

python train_generator.py --gpu_ids $gpu_ids --workers $workers --batch_size $batch_size --all_epochs $all_epochs \
                          --pre_epoch $pre_epoch --hdim $hdim --test_epoch $test_epoch --save_epoch $save_epoch \
                          --lambda_mmd $lambda_mmd --lambda_ip $lambda_ip --lambda_pair $lambda_pair \
                          --ip_model $ip_model --img_root $img_root --train_list $train_list --out_path $out_path \
                          --checkpoint_path $checkpoint_path --attack_type $attack_type --lambda_type $lambda_type\
                          --lambda_ort $lambda_ort\
                          | tee train_generator.log
