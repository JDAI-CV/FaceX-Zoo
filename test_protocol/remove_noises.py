"""
@author: JiXuan Xu, Jun Wang
@date: 20201012
@contact: jun21wangustc@gmail.com 
"""

# based on:
# https://github.com/deepinsight/insightface/blob/master/Evaluation/Megaface/remove_noises.py
# for 1M test, remove facescrub noise 25, megaface noise 686.

import os
import yaml
import numpy as np
import argparse
import shutil
import logging as logger

logger.basicConfig(level = logger.INFO, 
                   format = '%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt = '%Y-%m-%d %H:%M:%S')

feature_dim = 512

def remove_facescrub_noises(facescrub_noises_file, facescrub_feature_dir, facescrub_feature_outdir):
    """Remove the noise in facescrub.
    We use the class center of certain id as the feature of noise faces.

    Args:
        facescrub_noises_file(str): the path of facescrub noise list provided by deepglint.
        facescrub_feature_dir(str): thpe directory which contains the features of facescrub.
        facescrub_feature_outdir(str): the directory to save the clean features of facescrub.
    """
    noise_image2person_name = {}
    for line in open(facescrub_noises_file, 'r'):
        if line.startswith('#'):
            continue
        noise_image = line.strip()
        fname = noise_image.split('.')[0]
        person_name = fname[0 : fname.rfind('_')]
        noise_image2person_name[noise_image] = person_name
    logger.info('Total noise images in facescrub: %d.' % len(noise_image2person_name))
    person_name2center = {}
    noises = []
    for root, dirs, files in os.walk(facescrub_feature_dir):
        for feat_name in files:
            feat_name_ext = os.path.splitext(feat_name)[-1]
            if feat_name_ext == '.npy':
                feat_path = os.path.join(root, feat_name)
                assert(os.path.exists(feat_path))
                person_name = feat_path.split('/')[-2]
                image_name = feat_name[:-4] + '.jpg'
                cur_feature_outdir = os.path.join(facescrub_feature_outdir, person_name)
                if not os.path.exists(cur_feature_outdir):
                    os.makedirs(cur_feature_outdir)
                cur_feature_outpath = os.path.join(cur_feature_outdir, feat_name)
                if not image_name in noise_image2person_name:
                    cur_feature = np.load(feat_path)
                    np.save(cur_feature_outpath, cur_feature)
                    if not person_name in person_name2center:
                        person_name2center[person_name] = np.zeros((feature_dim,), dtype=np.float32)
                    person_name2center[person_name] += cur_feature
                else:
                    noises.append((person_name, feat_name))
    logger.info('Total noise images in current facescrub dir: %d.' % len(noises))
    for (person_name, feat_name) in noises:
        assert person_name in person_name2center
        center = person_name2center[person_name]
        center /= np.linalg.norm(center)
        cur_feature_outpath = os.path.join(facescrub_feature_outdir, person_name, feat_name)
        np.save(cur_feature_outpath, center)

def remove_megaface_noises(megaface_noises_file, megaface_feature_dir, megaface_feature_outdir):
    """Remove the noise in megaface.
    We set the feature of noise faces to zero vector, 
    since we use cos similarity as the distance metric.

    Args:
        megaface_noises_file(str): the path of megaface noise list provided by deepglint.
        megaface_feature_dir(str): thpe directory which contains the features of megaface.
        megaface_feature_outdir(str): the directory to save the clean features of megaface.
    """
    noise_image_set = set()
    for line in open(megaface_noises_file, 'r'):
        if line.startswith('#'):
            continue
        line = line.strip()
        _vec = line.split("\t")
        if len(_vec)>1:
            line = _vec[1]
        noise_image_set.add(line)
    logger.info('Total noise images in megaface: %d.' % len(noise_image_set))

    count_noises = 0
    for root, dirs, files in os.walk(megaface_feature_dir):
        for feat_name in files:
            feat_name_ext = os.path.splitext(feat_name)[-1]
            if feat_name_ext == '.npy':
                feat_path = os.path.join(root, feat_name)
                assert(os.path.exists(feat_path))
                id1 = feat_path.split('/')[-3]
                id2 = feat_path.split('/')[-2]
                image_name = feat_name[:-4] + '.jpg'
                cur_feature_outdir = os.path.join(megaface_feature_outdir, id1, id2)
                if not os.path.exists(cur_feature_outdir):
                    os.makedirs(cur_feature_outdir)
                cur_feature_outpath = os.path.join(cur_feature_outdir, feat_name)
                short_image_path = os.path.join(id1, id2, image_name)
                if not short_image_path in noise_image_set:
                    shutil.copyfile(feat_path, cur_feature_outpath)
                else:
                    cur_feature =  np.zeros((feature_dim,), dtype=np.float32)
                    np.save(cur_feature_outpath, cur_feature)
                    count_noises += 1
    logger.info('Total noise images in current megaface dir: %d.' % count_noises)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_conf_file', type=str,
                        help = "The path of data_conf.yaml.")
    parser.add_argument('--remove_facescrub_noise', type=int, 
                        help = "Remove facescrub noise or not, 1 for remove.")
    parser.add_argument('--remove_megaface_noise', type=int,
                        help = "Remove megaface noise or not, 1 for remove.")
    parser.add_argument('--facescrub_feature_dir', type=str, default='')
    parser.add_argument('--facescrub_feature_outdir', type=str, default='')
    parser.add_argument('--megaface_feature_dir', type=str, default='')
    parser.add_argument('--megaface_feature_outdir', type=str, default='')
    parser.add_argument('--masked_facescrub_feature_dir', type=str, default='')
    parser.add_argument('--masked_facescrub_feature_outdir', type=str, default='')
    args = parser.parse_args()
    with open(args.data_conf_file) as f:
        data_conf  = yaml.load(f)['MegaFace']
        facescrub_noises_file = data_conf['facescrub_noises_file']
        megaface_noises_file = data_conf['megaface_noises_file']
        megaface_mask = data_conf['megaface-mask']
    if args.remove_facescrub_noise == 1:
        remove_facescrub_noises(
            facescrub_noises_file, args.facescrub_feature_dir, args.facescrub_feature_outdir)
    if args.remove_megaface_noise == 1:
        remove_megaface_noises(
            megaface_noises_file, args.megaface_feature_dir, args.megaface_feature_outdir)
    if megaface_mask == 1:
        remove_facescrub_noises(
            facescrub_noises_file, args.masked_facescrub_feature_dir, args.masked_facescrub_feature_outdir)
