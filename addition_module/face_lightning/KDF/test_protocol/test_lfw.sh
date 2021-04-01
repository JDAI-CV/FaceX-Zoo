python test_lfw.py \
    --test_set 'LFW' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'ResNet' \
    --backbone_conf_file '/export/home/wangjun492/wj_armory/FaceX-Zoo/addition_module/face_lightning/KDF/training_mode/teacher_backbone_conf.yaml' \
    --batch_size 2048 \
    --model_path '/export/home/wangjun492/wj_armory/FaceX-Zoo/addition_module/face_lightning/KDF/training_mode/kd_training/pretrained_teacher'
