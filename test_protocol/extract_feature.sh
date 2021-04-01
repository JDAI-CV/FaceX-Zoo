python extract_feature.py \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'MobileFaceNet' \
    --backbone_conf_file 'backbone_conf.yaml' \
    --batch_size 10240 \
    --model_path 'Epoch_17.pt' \
    --feats_root 'feats' 
