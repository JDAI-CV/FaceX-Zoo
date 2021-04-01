mkdir 'log'
python train_ft.py \
    --data_root '/export2/wangjun492/face_database/facex-zoo/private_file/train_data/deepglint/msra_crop' \
    --train_file '/export2/wangjun492/face_database/facex-zoo/private_file/train_data/deepglint/msceleb_deepglint_train_file.txt' \
    --teacher_backbone_type 'ResNet' \
    --teacher_backbone_conf_file '../teacher_backbone_conf.yaml' \
    --student_backbone_type 'MobileFaceNet' \
    --student_backbone_conf_file '../student_backbone_conf.yaml' \
    --head_type 'MV-Softmax' \
    --head_conf_file '../head_conf.yaml' \
    --loss_type 'SoftTarget' \
    --loss_conf_file '../loss_conf.yaml' \
    --lr 0.1 \
    --lambda_kd 0.1 \
    --out_dir 'out_dir' \
    --epoches 18 \
    --step '10, 13, 16' \
    --print_freq 200 \
    --save_freq 3000 \
    --batch_size 512 \
    --momentum 0.9 \
    --log_dir 'log' \
    --tensorboardx_logdir 'mv-hrnet' \
    --pretrained_teacher 'pretrained_teacher/Epoch_4.pt' \
    2>&1 | tee log/log.log
