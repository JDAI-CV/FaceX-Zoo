import os

def write_config_into_log(cfg):
    attrs = dir(cfg)
    no_print_attributes = ['ckpt_root_dir', 'default_seed', 'get_lr',
                           'tb_dump_interval', 'iter_per_epoch', 'program_name',
                           'this_model_dir', 'train_data_num_thread', 'train_dp_name',
                           'workspace_limit', 'train_dataset', 'val_dataset']
    for attr in attrs:
        if not (('__' in attr) or (attr in no_print_attributes)):
            with open( os.path.join(cfg.output_dir, 'config.txt'), 'a+') as f:
                f.write('{0},{1}\n'.format(attr, getattr(cfg, attr)))

