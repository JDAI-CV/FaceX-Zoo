# Requirements
* python == 3.8.8
* pytorch == 1.7.1
* torchvision == 0.8.2
* timm == 0.3.2
* apex == 0.1

# Training script
* Swin-T
```
export OMP_NUM_THREADS=4
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1324 train.py --data_root 'train_data/msra_crop' --train_file 'train_data/msceleb_deepglint_train_file.txt' --backbone_type 'SwinTransformer' --backbone_conf_file '../backbone_conf.yaml' --head_type 'MV-Softmax' --head_conf_file '../head_conf.yaml' --lr 5e-4 --out_dir 'out_dir' --epoches 18 --warm_up_epoches 1 --print_freq 200 --save_freq 3000 --batct_size 128  --log_dir 'log' --tensorboardx_logdir 'mv-swin' 2>&1 | tee log.log
```
