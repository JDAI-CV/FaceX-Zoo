"""
@author: Haoran Jiang, Jun Wang
@date: 20201013
@contact: jun21wangustc@gmail.com
"""

import os
import sys
import numpy as np
    
class LFWEvaluator(object):
    """Implementation of LFW test protocal.
    
    Attributes:
        data_loader(object): a test data loader.
        pair_list(list): the pair list given by PairsParser.
        feature_extractor(object): a feature extractor.
    """
    def __init__(self, data_loader, pairs_parser_factory, feature_extractor):
        """Init LFWEvaluator.

        Args:
            data_loader(object): a test data loader. 
            pairs_parser_factory(object): factory to produce the parser to parse test pairs list.
            pair_list(list): the pair list given by PairsParser.
            feature_extractor(object): a feature extractor.
        """
        self.data_loader = data_loader
        pairs_parser = pairs_parser_factory.get_parser()
        self.pair_list = pairs_parser.parse_pairs()
        self.feature_extractor = feature_extractor

    def test(self, model):
        image_name2feature = self.feature_extractor.extract_online(model, self.data_loader)
        mean, std = self.test_one_model(self.pair_list, image_name2feature)
        return mean, std

    def test_one_model(self, test_pair_list, image_name2feature, is_normalize = True):
        """Get the accuracy of a model.
        
        Args:
            test_pair_list(list): the pair list given by PairsParser. 
            image_name2feature(dict): the map of image name and it's feature.
            is_normalize(bool): wether the feature is normalized.

        Returns:
            mean: estimated mean accuracy.
            std: standard error of the mean.
        """
        subsets_score_list = np.zeros((10, 600), dtype = np.float32)
        subsets_label_list = np.zeros((10, 600), dtype = np.int8)
        for index, cur_pair in enumerate(test_pair_list):
            cur_subset = index // 600
            cur_id = index % 600
            image_name1 = cur_pair[0]
            image_name2 = cur_pair[1]
            label = cur_pair[2]
            subsets_label_list[cur_subset][cur_id] = label
            feat1 = image_name2feature[image_name1]
            feat2 = image_name2feature[image_name2]
            if not is_normalize:
                feat1 = feat1 / np.linalg.norm(feat1)
                feat2 = feat2 / np.linalg.norm(feat2)
            cur_score = np.dot(feat1, feat2)
            subsets_score_list[cur_subset][cur_id] = cur_score

        subset_train = np.array([True] * 10)
        accu_list = []
        for subset_idx in range(10):
            test_score_list = subsets_score_list[subset_idx]
            test_label_list = subsets_label_list[subset_idx]
            subset_train[subset_idx] = False
            train_score_list = subsets_score_list[subset_train].flatten()
            train_label_list = subsets_label_list[subset_train].flatten()
            subset_train[subset_idx] = True
            best_thres = self.getThreshold(train_score_list, train_label_list)
            positive_score_list = test_score_list[test_label_list == 1]
            negtive_score_list = test_score_list[test_label_list == 0]
            true_pos_pairs = np.sum(positive_score_list > best_thres)
            true_neg_pairs = np.sum(negtive_score_list < best_thres)
            accu_list.append((true_pos_pairs + true_neg_pairs) / 600)
        mean = np.mean(accu_list)
        std = np.std(accu_list, ddof=1) / np.sqrt(10) #ddof=1, division 9.
        return mean, std

    def getThreshold(self, score_list, label_list, num_thresholds=1000):
        """Get the best threshold by train_score_list and train_label_list.
        Args:
            score_list(ndarray): the score list of all pairs.
            label_list(ndarray): the label list of all pairs.
            num_thresholds(int): the number of threshold that used to compute roc.
        Returns:
            best_thres(float): the best threshold that computed by train set.
        """
        pos_score_list = score_list[label_list == 1]
        neg_score_list = score_list[label_list == 0]
        pos_pair_nums = pos_score_list.size
        neg_pair_nums = neg_score_list.size
        score_max = np.max(score_list)
        score_min = np.min(score_list)
        score_span = score_max - score_min
        step = score_span / num_thresholds
        threshold_list = score_min +  step * np.array(range(1, num_thresholds + 1)) 
        fpr_list = []
        tpr_list = []
        for threshold in threshold_list:
            fpr = np.sum(neg_score_list > threshold) / neg_pair_nums
            tpr = np.sum(pos_score_list > threshold) /pos_pair_nums
            fpr_list.append(fpr)
            tpr_list.append(tpr)
        fpr = np.array(fpr_list)
        tpr = np.array(tpr_list)
        best_index = np.argmax(tpr-fpr)
        best_thres = threshold_list[best_index]
        return  best_thres
