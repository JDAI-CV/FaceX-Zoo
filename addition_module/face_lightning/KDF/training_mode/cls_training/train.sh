mkdir 'log'
python train.py \
    --data_root '/home/wangjun492/wj_data/face_database/facex-zoo/msra_crop' \
    --train_file '/home/wangjun492/wj_data/face_database/facex-zoo/msceleb_deepglint_train_file.txt' \
    --backbone_type 'ResNet' \
    --backbone_conf_file '../student_backbone_conf.yaml' \
    --head_type 'MV-Softmax' \
    --head_conf_file '../head_conf.yaml' \
    --lr 0.1 \
    --out_dir 'out_dir' \
    --epoches 18 \
    --step '10, 13, 16' \
    --print_freq 200 \
    --save_freq 3000 \
    --batch_size 512 \
    --momentum 0.9 \
    --log_dir 'log' \
    --tensorboardx_logdir 'mv-resnet' \
    2>&1 | tee log/log.log
