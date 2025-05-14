python remove_noises.py \
    --data_conf_file 'data_conf.yaml' \
    --remove_facescrub_noise 1 \
    --remove_megaface_noise 1 \
    --facescrub_feature_dir 'feats/facescrub_crop' \
    --facescrub_feature_outdir 'feats_clean/facescrub_crop' \
    --megaface_feature_dir 'feats/megaface_crop' \
    --megaface_feature_outdir 'feats_clean/megaface_crop' \
    --masked_facescrub_feature_dir 'feats/masked_facescrub_crop' \
    --masked_facescrub_feature_outdir 'feats_clean/masked_facescrub_crop'
