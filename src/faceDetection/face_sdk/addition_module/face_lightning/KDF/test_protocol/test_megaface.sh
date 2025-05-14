python test_megaface.py \
    --data_conf_file 'data_conf.yaml' \
    --max_rank 10 \
    --facescrub_feature_dir 'feats_concat/facescrub_crop' \
    --megaface_feature_dir 'feats_concat/megaface_crop' \
    --masked_facescrub_feature_dir 'feats_concat/masked_facescrub_crop' \
    --is_concat 1
