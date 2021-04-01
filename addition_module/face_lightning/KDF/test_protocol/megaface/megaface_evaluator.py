"""
@author: Jun Wang
@date: 20201012
@contact: jun21wangustc@gmail.com
"""
import os
import logging as logger
import json
import numpy as np
from abc import ABCMeta, abstractmethod
from prettytable import PrettyTable

logger.basicConfig(level = logger.INFO,
                   format = '%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt = '%Y-%m-%d %H:%M:%S')

class MegaFaceEvaluator(metaclass=ABCMeta):
    """The base class for MegaFace test protocal.
    Load and parse the json files and features.

    Attributes:
        facescrub_feature_dir(str): the directory whcih contains the features of facescurb.
        masked_facescrub_feature_dir(str): the directory which contains the features of masked facescurb.
        megaface_feature_dir(str): the directory which contains the features of megaface.

        megaface_path_list(list): the path list of megaface files.
        facescrub_path_list(list): the path list of facescrub files.
        facescrub_id_list(list): the id list of facescrub files, corresponding to facescrub_path_list. 
        facescrub_id2index_list(dict): the index in facescrub_path_list of every id.

        megaface_feature_list(ndarray): the numpy array of megaface features, the shape is N×feat_dim.
        facescrub_feature_list(ndarray): the numpy array of facescrub features, the shape is N×feat_dim.
        facescrub_id2feature_list(dict): the feature list of every id in facescrub.
    """
    def __init__(self, facescrub_json_list, megaface_json_list, 
                 facescrub_feature_dir, megaface_feature_dir, masked_facescrub_feature_dir, is_concat = False):
        """Init MegaFaceEvaluator by some initial files.

        Args:
            facescrub_json_list(str): the facescrub list provided by official.
            megaface_json_list(str): the megaface list provided by official.
            facescrub_feature_dir(str): the directory whcih contains the features of facescur.
            masked_facescrub_feature_dir(str): the directory which contains the features of masked facescurb.
            megaface_feature_dir(str): the directory which contains the features of megaface.
            is_concat(bool): if the feature is concated by two nomalized features.
        """
        self.is_concat = is_concat
        self.facescrub_feature_dir = facescrub_feature_dir
        self.masked_facescrub_feature_dir = masked_facescrub_feature_dir
        self.megaface_feature_dir = megaface_feature_dir
        megaface_json = json.load(open(megaface_json_list))
        self.megaface_path_list = megaface_json['path']
        facescrub_json = json.load(open(facescrub_json_list))
        self.facescrub_path_list = facescrub_json['path']
        self.facescrub_id_list = facescrub_json['id']
        self.facescrub_id2index_list = {} #index为facescrub_path_list中的下标
        for index, facescrub_id in enumerate(self.facescrub_id_list):
            if not facescrub_id in self.facescrub_id2index_list:
                self.facescrub_id2index_list[facescrub_id] = []
            self.facescrub_id2index_list[facescrub_id].append(index)
        logger.info('Megaface image nums: %d.' % len(self.megaface_path_list))
        logger.info('Facescrub id nums: %d, image nums: %d.' % 
              (len(self.facescrub_id2index_list), len(self.facescrub_path_list)))
        logger.info('Loading magaface feature ...')
        self.megaface_feature_list = self.get_megaface_feature_list()
        logger.info('Loading facescrub feature ...')
        self.facescrub_feature_list = self.get_facescrub_feature_list()
        logger.info('Loading down.')
        self.facescrub_id2feature_list = self.get_facescrub_id2feature_list()

    def argNmax(self, score_mat, N, axis = None):
        """Get the index/indexes of the Nth largest element.
        Suppose the shape of score_mat is R*C, axis=1 mean that 
        the indexes of the Nth largest element of every row will be returned.

        Args:
            score_mat(numpy array): the input mat or array.

        Returns:
            The index/indexes of the Nth largest element.
        """
        if axis is None: 
            return np.argpartition(score_mat.ravel(), -N)[-N]
        else:
            return np.take(np.argpartition(score_mat, -N, axis=axis), -N, axis=axis)

    def get_megaface_feature_list(self):
        """Get the feature list of megaface.

        Returns:
            A numpy array, the shape is N * feat_dim, N is the number of images in megaface.
        """
        megaface_feature_list = []
        for megaface_file_path in self.megaface_path_list:
            cur_megaface_file_path = megaface_file_path.replace('.jpg', '.npy')
            cur_megaface_file = os.path.join(self.megaface_feature_dir, cur_megaface_file_path)
            assert(os.path.exists(cur_megaface_file))
            cur_megaface_feat = np.load(cur_megaface_file)
            megaface_feature_list.append(cur_megaface_feat)
        return np.array(megaface_feature_list)

    def get_facescrub_feature_list(self):
        """Get the feature list of facescrub.

        Returns:
            A numpy array, the shape is N * feat_dim, N is the number of images in facescrub.
        """
        facescrub_feature_list = []
        for facescrub_path in self.facescrub_path_list:
            facescrub_path = facescrub_path.replace('.jpg', '.npy')
            facescrub_file = os.path.join(self.facescrub_feature_dir, facescrub_path)
            assert(os.path.exists(facescrub_file))
            facescrub_feat = np.load(facescrub_file)
            facescrub_feature_list.append(facescrub_feat)
        return np.array(facescrub_feature_list)

    def get_facescrub_id2feature_list(self):
        """Get the feature list of each id in facescrub.

        Returns:
           A dict，key is the id in facescrub，value is the feature list of this id.
        """
        facescrub_id2feature_list = {}
        for facescrub_id, cur_feature in zip(self.facescrub_id_list, self.facescrub_feature_list):
            if not facescrub_id in facescrub_id2feature_list:
                facescrub_id2feature_list[facescrub_id] = []
            facescrub_id2feature_list[facescrub_id].append(cur_feature)
        facescrub_id2feature_array = {}
        for facescrub_id, feature_list in facescrub_id2feature_list.items():
            facescrub_id2feature_array[facescrub_id] = np.array(feature_list)
        return facescrub_id2feature_array

    def test_cmc(self, max_rank):
        """Get the cmc curve for megaface
        Compute the accuracy of rank 1 to rank max_rank 

        Args:
            max_rank(int): the max rank to compute.
        """
        pretty_tabel = PrettyTable(["rank", "accuracy"])
        for cur_rank in range(max_rank):
            cur_rank += 1
            cur_accuracy = self.get_rankN_accuracy(cur_rank)
            logger.info('Rank %d accuracy: %f.' % (cur_rank, cur_accuracy))
            pretty_tabel.add_row([cur_rank, cur_accuracy])
        print(pretty_tabel)

    @abstractmethod
    def get_rankN_accuracy(self, cur_rank):
        """Get the rank N accuracy

        Args:
            cur_rank(int): current rank, cur_rank=1 for rank1 accuracy.

        Returns:
            accuracy(float): the accuracy for current rank.
        """
        pass
    
class CommonMegaFaceEvaluator(MegaFaceEvaluator):
    """The common MegaFace test protocal.
    Python implementation of megaface test protocal, 
    the same as the official implementation by .bin.

    Attributes:
        facescrub_feature_dir(str): inherit from the parent class.
        masked_facescrub_feature_dir(str): inherit from the parent class.
        megaface_feature_dir(str): inherit from the parent class.
        megaface_path_list(list): inherit from the parent class.
        facescrub_path_list(list): inherit from the parent class.
        facescrub_id2index_list(dict): inherit from the parent class.
        megaface_feature_list(ndarray): inherit from the parent class.
        facescrub_feature_list(ndarray): inherit from the parent class.
        facescrub_id2feature_list(dict): inherit from the parent class.

        facescrub_megaface_score(ndarray): facescrub to megaface scores, N*M matrix, 
                                           N is the images of facescrub, M is the images of megaface.
        facescrub_id2intra_score(dict): the intra-class scores of facescrub.
    """
    def __init__(self, facescrub_json_list, megaface_json_list, 
                 facescrub_feature_dir, megaface_feature_dir, is_concat):
        """Init CommonMegaFaceEvaluator by some initial files. 

        Args:
            Please refer to the init method in parent class.
        """
        super().__init__(facescrub_json_list, megaface_json_list, 
                         facescrub_feature_dir, megaface_feature_dir, None, is_concat)
        self.facescrub_id2intra_score = {}
        for facescrub_id, feature_list in self.facescrub_id2feature_list.items(): 
            cur_facescrub_score = np.dot(feature_list, feature_list.T) 
            if self.is_concat:
                cur_facescrub_score = 1/2 * cur_facescrub_score
            self.facescrub_id2intra_score[facescrub_id] = cur_facescrub_score
        cur_facescurb_megaface_score = \
            np.dot(self.facescrub_feature_list, self.megaface_feature_list.T)
        if self.is_concat:
            cur_facescurb_megaface_score = 1/2 * cur_facescurb_megaface_score
        self.facescrub_megaface_score = cur_facescurb_megaface_score

    def get_rankN_accuracy(self, cur_rank):
        facescrub_rankN_galley_index = \
            self.argNmax(self.facescrub_megaface_score, cur_rank, axis = 1)
        facescrub_rankN_gallery_score = []
        for index in range(len(self.facescrub_path_list)):
            rankN_index = facescrub_rankN_galley_index[index]
            rankN_score = self.facescrub_megaface_score[index][rankN_index]
            facescrub_rankN_gallery_score.append(rankN_score)
        count_all = 0
        TP = 0
        for facescrub_id, intra_score in self.facescrub_id2intra_score.items():# N people
            M, _ = intra_score.shape # M photos 
            for image_index in range(M): # Add Mi to megaface and test other M-1 photos
                for facescrub_index in range(M):
                    if facescrub_index == image_index:
                        continue
                    cur_positive_score = intra_score[image_index][facescrub_index]
                    cur_facescrub_global_index = \
                        self.facescrub_id2index_list[facescrub_id][facescrub_index]
                    cur_megaface_rankN_score = \
                        facescrub_rankN_gallery_score[cur_facescrub_global_index]
                    if cur_positive_score >= cur_megaface_rankN_score:
                        TP += 1
                    count_all += 1
        accuracy = TP / count_all
        return accuracy

class MaskedMegaFaceEvaluator(MegaFaceEvaluator):
    """The masked MegaFace test protocal.
    The feature of every probe is extracted from the masked face.
    The feature of every gallery is extracted from the common face.

    Attributes:
        facescrub_feature_dir(str): inherit from the parent class.
        masked_facescrub_feature_dir(str): inherit from the parent class.
        megaface_feature_dir(str): inherit from the parent class.
        megaface_path_list(list): inherit from the parent class.
        facescrub_path_list(list): inherit from the parent class.
        facescrub_id2index_list(dict): inherit from the parent class.
        megaface_feature_list(ndarray): inherit from the parent class.
        facescrub_feature_list(ndarray): inherit from the parent class.
        facescrub_id2feature_list(dict): inherit from the parent class.

        masked_facescrub_feature_list(ndarray): the numpy array of masked facescrub features, 
                                                the shape is N×feat_dim.
        masked_facescrub_id2feature_list(dict): the feature list of every id in facescrub.
        masked_facescrub_megaface_score(ndarray): masked facescrub to megaface scores, N*M matrix, 
                                           N is the images of facescrub, M is the images of megaface.
        masked_facescrub_id2intra_score(dict): the intra-class scores of facescrub.
    """
    def __init__(self, facescrub_json_list, megaface_json_list, 
                 facescrub_feature_dir, megaface_feature_dir, masked_facescrub_feature_dir, is_concat):
        """Init CommonMegaFaceEvaluator by some initial files. 

        Args:
            Please refer to the init method in parent class.
        """
        super().__init__(facescrub_json_list, megaface_json_list, 
                         facescrub_feature_dir, megaface_feature_dir, masked_facescrub_feature_dir, is_concat)
        self.masked_facescrub_feature_list = self.get_masked_facescrub_feature_list()
        self.masked_facescrub_id2feature_list = self.get_masked_facescrub_id2feature_list()
        self.masked_facescrub_id2intra_score = {}
        for facescrub_id, masked_feature_list in self.masked_facescrub_id2feature_list.items():
            feature_list = self.facescrub_id2feature_list[facescrub_id]
            cur_masked_facescrub_score = np.dot(feature_list, masked_feature_list.T)
            if is_concat:
                cur_masked_facescrub_score = 1/2 * cur_masked_facescrub_score
            self.masked_facescrub_id2intra_score[facescrub_id] = cur_masked_facescrub_score
        cur_masked_facescrub_megaface_score = \
            np.dot(self.masked_facescrub_feature_list, self.megaface_feature_list.T)
        if is_concat:
            cur_masked_facescrub_megaface_score = 1/2 * cur_masked_facescrub_megaface_score
        self.masked_facescrub_megaface_score = cur_masked_facescrub_megaface_score

    def get_masked_facescrub_feature_list(self):
        """Get the feature list of masked facescrub.

        Returns:
            A numpy array, the shape is N * feat_dim, N is the number of images in facescrub.
        """
        masked_facescrub_feature_list = []
        for facescrub_path in self.facescrub_path_list:
            facescrub_path = facescrub_path.replace('.jpg', '.npy')
            facescrub_file = os.path.join(self.masked_facescrub_feature_dir, facescrub_path)
            assert(os.path.exists(facescrub_file))
            facescrub_feat = np.load(facescrub_file)
            masked_facescrub_feature_list.append(facescrub_feat)
        return np.array(masked_facescrub_feature_list)

    def get_masked_facescrub_id2feature_list(self):
        """Get the feature list of each id in masked facescrub.

        Returns:
            A dict，key is the id in masked facescrub，value is the feature list of this id.
        """
        masked_facescrub_id2feature_list = {}
        for facescrub_id, cur_feature in zip(self.facescrub_id_list, self.masked_facescrub_feature_list):
            if not facescrub_id in masked_facescrub_id2feature_list:
                masked_facescrub_id2feature_list[facescrub_id] = []
            masked_facescrub_id2feature_list[facescrub_id].append(cur_feature)
        masked_facescrub_id2feature_array = {}
        for facescrub_id, feature_list in masked_facescrub_id2feature_list.items():
            masked_facescrub_id2feature_array[facescrub_id] = np.array(feature_list)
        return masked_facescrub_id2feature_array

    def get_rankN_accuracy(self, cur_rank):
        facescrub_rankN_galley_index = \
            self.argNmax(self.masked_facescrub_megaface_score, cur_rank, axis = 1)
        facescrub_rankN_gallery_score = []
        for index in range(len(self.facescrub_path_list)):
            rankN_index = facescrub_rankN_galley_index[index]
            rankN_score = self.masked_facescrub_megaface_score[index][rankN_index]
            facescrub_rankN_gallery_score.append(rankN_score)
        count_all = 0
        TP = 0
        for facescrub_id, masked_intra_score in self.masked_facescrub_id2intra_score.items():# N people
            M, _ = masked_intra_score.shape # M photos 
            for image_index in range(M): # Add Mi to megaface and test other M-1 photos
                for facescrub_index in range(M):
                    if facescrub_index == image_index:
                        continue
                    cur_positive_score = masked_intra_score[image_index][facescrub_index]
                    cur_facescrub_global_index = self.facescrub_id2index_list[facescrub_id][facescrub_index]
                    cur_megaface_rankN_score = facescrub_rankN_gallery_score[cur_facescrub_global_index]
                    if cur_positive_score >= cur_megaface_rankN_score:
                        TP += 1
                    count_all += 1
        accuracy = TP / count_all
        return accuracy
