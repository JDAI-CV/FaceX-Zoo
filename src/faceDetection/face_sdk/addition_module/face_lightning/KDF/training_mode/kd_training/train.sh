mkdir 'log'
python3 train.py \
    --data_root '/home/wangjun492/wj_data/face_database/facex-zoo/msra_crop' \
    --train_file '/home/wangjun492/wj_data/face_database/facex-zoo/msceleb_deepglint_train_file.txt' \
    --teacher_backbone_type 'ResNet' \
    --teacher_backbone_conf_file '../teacher_backbone_conf.yaml' \
    --student_backbone_type 'ResNet' \
    --student_backbone_conf_file '../student_backbone_conf.yaml' \
    --head_type 'MV-Softmax' \
    --head_conf_file '../head_conf.yaml' \
    --loss_type 'RKD' \
    --loss_conf_file '../loss_conf.yaml' \
    --lr 0.1 \
    --lambda_kd 1. \
    --out_dir 'out_dir' \
    --epoches 18 \
    --step '10, 13, 16' \
    --print_freq 200 \
    --save_freq 3000 \
    --batch_size 512 \
    --momentum 0.9 \
    --log_dir 'log' \
    --tensorboardx_logdir 'mv-ft' \
    --pretrained_teacher 'pretrained_teacher/Epoch_17.pt' \
    2>&1 | tee log/log.log
