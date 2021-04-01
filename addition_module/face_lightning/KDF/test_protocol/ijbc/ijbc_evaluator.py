"""
@author: Jun Wang
@date: 20210308
@contact: jun21wangustc@gmail.com
"""

# based on:
# https://github.com/deepinsight/insightface/tree/master/evaluation/IJB

import numpy as np
from numpy import matlib
from prettytable import PrettyTable
from sklearn.metrics import roc_curve

class IJBCEvaluator(object):
    """Implementation of IJBC test protocal.
    """
    def __init__(self, template_media_list, template_pair_list, image_list, data_loader, feature_extractor):
        """Init IJBCEvaluator.
        
        Args:
            template_media_list(str): the path of 'ijbc_face_tid_mid.txt'
            template_pair_list(str): the path of 'ijbc_template_pair_label.txt '
            image_list(str): the path of 'img_list.txt'
            data_loader(object): a test data loader.
            feature_extractor(object): a feature extractor.            
        """
        templates = []
        medias = []
        template_media_list_buf = open(template_media_list)
        line = template_media_list_buf.readline().strip()
        while line:
            image_name, tid, mid = line.split(' ')
            templates.append(int(tid))
            medias.append(int(mid))
            line = template_media_list_buf.readline().strip()
        self.templates = np.array(templates)
        self.medias = np.array(medias)

        template1 = []
        template2 = []
        label = []
        template_pair_list_buf = open(template_pair_list)
        line = template_pair_list_buf.readline().strip()
        while line:
            t1, t2, cur_label = line.split(' ')
            template1.append(int(t1))
            template2.append(int(t2))
            label.append(int(cur_label))
            line = template_pair_list_buf.readline().strip()
        self.template1 = np.array(template1)
        self.template2 = np.array(template2)
        self.label = np.array(label)

        self.image_list = []
        faceness_scores = []
        image_list_buf = open(image_list)
        line = image_list_buf.readline().strip()
        while line:
            self.image_list.append(line.split(' ')[0])
            faceness_scores.append(float(line.split(' ')[-1]))
            line = image_list_buf.readline().strip()
        self.faceness_scores = np.array(faceness_scores)

        self.data_loader = data_loader
        self.feature_extractor = feature_extractor

    def verification(self, template_norm_feats, unique_templates):
        template2id = np.zeros((max(unique_templates)+1, 1), dtype=int)
        for count_template, uqt in enumerate(unique_templates):
            template2id[uqt] = count_template
        score = np.zeros((len(self.template1),))   # save cosine distance between pairs 
        total_pairs = np.array(range(len(self.template1)))
        batchsize = 100000 # small batchsize instead of all pairs in one batch due to the memory limiation
        sublists = [total_pairs[i:i + batchsize] for i in range(0, len(self.template1), batchsize)]
        total_sublists = len(sublists)
        for c, s in enumerate(sublists):
            feat1 = template_norm_feats[template2id[self.template1[s]]]
            feat2 = template_norm_feats[template2id[self.template2[s]]]
            similarity_score = np.sum(feat1 * feat2, -1)
            score[s] = similarity_score.flatten()
            if c % 10 == 0:
                print('Finish {}/{} pairs.'.format(c, total_sublists))
        return score

    def image2template_feature(self, img_feats):
        unique_templates = np.unique(self.templates)
        template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))
        for count_template, uqt in enumerate(unique_templates):
            (ind_t,) = np.where(self.templates == uqt)
            face_norm_feats = img_feats[ind_t]
            face_medias = self.medias[ind_t]
            unique_medias, unique_media_counts = np.unique(face_medias, return_counts=True)
            media_norm_feats = []
            for u,ct in zip(unique_medias, unique_media_counts):
                (ind_m,) = np.where(face_medias == u)
                if ct == 1:
                    media_norm_feats += [face_norm_feats[ind_m]]
                else: # image features from the same video will be aggregated into one feature
                    media_norm_feats += [np.mean(face_norm_feats[ind_m], 0, keepdims=True)]
            media_norm_feats = np.array(media_norm_feats)
            # media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))
            template_feats[count_template] = np.sum(media_norm_feats, 0)
            if count_template % 2000 == 0: 
                print('Finish Calculating {} template features.'.format(count_template))
        template_norm_feats = template_feats / np.sqrt(np.sum(template_feats ** 2, -1, keepdims=True))
        return template_norm_feats, unique_templates

    def test(self, model, use_detector_score=True):
        fpr_list = [1e-6, 1e-5, 1e-4,1e-3, 1e-2, 1e-1]

        image_name2feature = self.feature_extractor.extract_online(model, self.data_loader)
        feature_list = []
        for image_name in self.image_list:
            feature_list.append(image_name2feature[image_name])
        feature_list = np.array(feature_list).astype(np.float32)

        #feature_list = np.load('/export2/wangjun492/face_database/facex-zoo/private_file/test_data/ijbc/img_feats.npy')
        if use_detector_score:
            feature_list = feature_list * matlib.repmat(self.faceness_scores[:,np.newaxis], 1, feature_list.shape[1])
        template_norm_feats, unique_templates = self.image2template_feature(feature_list)
        score = self.verification(template_norm_feats, unique_templates)
        fpr, tpr, _ = roc_curve(self.label, score)
        fpr = np.flipud(fpr)
        tpr = np.flipud(tpr) # select largest tpr at same fpr

        tpr_list = []
        for fpr_iter in np.arange(len(fpr_list)):
            _, min_index = min(list(zip(abs(fpr-fpr_list[fpr_iter]), range(len(fpr)))))
            tpr_list.append('%.4f' % tpr[min_index])
        return tpr_list
