mkdir 'log'
python train.py \
    --data_root '/export2/wangjun492/face_database/facex-zoo/private_file/train_data/deepglint/msra_crop' \
    --train_file '/export2/wangjun492/face_database/facex-zoo/share_file/train_data/MS-Celeb-1M-v1c-r-shallow_train_list.txt' \
    --backbone_type 'MobileFaceNet' \
    --backbone_conf_file '../backbone_conf.yaml' \
    --head_type 'SST_Prototype' \
    --head_conf_file '../head_conf.yaml' \
    --lr 0.1 \
    --out_dir 'out_dir' \
    --epoches 250 \
    --step '150,200,230' \
    --print_freq 100 \
    --batch_size 512 \
    --momentum 0.9 \
    --alpha 0.999 \
    --log_dir 'log' \
    --tensorboardx_logdir 'sst_mobileface' \
    --save_freq 10 \
    --evaluate \
    --test_set 'LFW' \
    --test_data_conf_file '/export/home/wangjun492/wj_armory/FaceX-Zoo/test_protocol/data_conf.yaml' \
    2>&1 | tee log/log.log
