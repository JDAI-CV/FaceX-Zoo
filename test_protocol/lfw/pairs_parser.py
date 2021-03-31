""" 
@author: Jixuan Xu, Jun Wang
@date: 20201015
@contact: jun21wangustc@gmail.com
""" 

import scipy.io as scio
from abc import ABCMeta, abstractmethod

class PairsParser(metaclass=ABCMeta):
    """Parse the pair list for lfw based protocol.
    Because the official pair list for different dataset(lfw, cplfw, calfw ...) is different, 
    we need different method to parse the pair list of different dataset.
    
    Attributes:
        pairs_file(str): the path of the pairs file that was released by official.
    """
    def __init__(self, pairs_file):
        """Init PairsParser
            
        Args:
            pairs_file(str): the path of the pairs file that was released by official.
        """
        self.pairs_file = pairs_file
    def parse_pairs(self):
        """The method for parsing pair list.
        """
        pass

class LFW_PairsParser(PairsParser):
    """The pairs parser for lfw.
    """
    def parse_pairs(self):
        test_pair_list = []
        pairs_file_buf = open(self.pairs_file)
        line = pairs_file_buf.readline() # skip first line
        line = pairs_file_buf.readline().strip()
        while line:
            line_strs = line.split('\t')
            if len(line_strs) == 3:
                person_name = line_strs[0]
                image_index1 = line_strs[1]
                image_index2 = line_strs[2]
                image_name1 = person_name + '/' + person_name + '_' + image_index1.zfill(4) + '.jpg'
                image_name2 = person_name + '/' + person_name + '_' + image_index2.zfill(4) + '.jpg'
                label = 1
            elif len(line_strs) == 4:
                person_name1 = line_strs[0]
                image_index1 = line_strs[1]
                person_name2 = line_strs[2]
                image_index2 = line_strs[3]
                image_name1 = person_name1 + '/' + person_name1 + '_' + image_index1.zfill(4) + '.jpg'
                image_name2 = person_name2 + '/' + person_name2 + '_' + image_index2.zfill(4) + '.jpg'
                label = 0
            else:
                raise Exception('Line error: %s.' % line)
            test_pair_list.append((image_name1, image_name2, label))
            line = pairs_file_buf.readline().strip()
        return test_pair_list

class RFW_PairsParser(PairsParser):
    """The pairs parser for lfw.
    """
    def parse_pairs(self):
        test_pair_list = []
        pairs_file_buf = open(self.pairs_file)
        line = pairs_file_buf.readline().strip()
        while line:
            line_strs = line.split('\t')
            if len(line_strs) == 3:
                person_name = line_strs[0]
                image_index1 = line_strs[1]
                image_index2 = line_strs[2]
                image_name1 = person_name + '/' + person_name + '_' + image_index1.zfill(4) + '.jpg'
                image_name2 = person_name + '/' + person_name + '_' + image_index2.zfill(4) + '.jpg'
                label = 1
            elif len(line_strs) == 4:
                person_name1 = line_strs[0]
                image_index1 = line_strs[1]
                person_name2 = line_strs[2]
                image_index2 = line_strs[3]
                image_name1 = person_name1 + '/' + person_name1 + '_' + image_index1.zfill(4) + '.jpg'
                image_name2 = person_name2 + '/' + person_name2 + '_' + image_index2.zfill(4) + '.jpg'
                label = 0
            else:
                raise Exception('Line error: %s.' % line)
            test_pair_list.append((image_name1, image_name2, label))
            line = pairs_file_buf.readline().strip()
        return test_pair_list

class CPLFW_PairsParser(PairsParser):
    """The pairs parser for cplfw.
    """
    def parse_pairs(self):        
        pair_list = []
        pairs_file_buf = open(self.pairs_file)
        line1 = pairs_file_buf.readline().strip()
        while line1:
            line2 = pairs_file_buf.readline().strip()
            image_name1 = line1.split(' ')[0]
            image_name2 = line2.split(' ')[0]
            label = line1.split(' ')[1]
            pair_list.append((image_name1, image_name2, int(label)))
            line1 = pairs_file_buf.readline().strip()
        assert(len(pair_list) == 6000)
        test_pair_list = []
        positive_start = 0 # 0-2999
        negtive_start = 3000 # 3000 - 5999
        for set_idx in range(10):
            positive_index = positive_start + 300 * set_idx
            negtive_index = negtive_start + 300 * set_idx
            cur_positive_pair_list = pair_list[positive_index : positive_index + 300]
            cur_negtive_pair_list = pair_list[negtive_index : negtive_index + 300]
            test_pair_list.extend(cur_positive_pair_list)
            test_pair_list.extend(cur_negtive_pair_list)
        return test_pair_list

class CALFW_PairsParser(PairsParser):
    """The pairs parser for calfw.
    """
    def parse_pairs(self):
        pair_list = []
        pairs_file_buf = open(self.pairs_file)
        line1 = pairs_file_buf.readline().strip()
        while line1:
            line2 = pairs_file_buf.readline().strip()
            image_name1 = line1.split(' ')[0]
            image_name2 = line2.split(' ')[0]
            label = int(line1.split(' ')[1])
            if label != 0:
                label = 1
            pair_list.append((image_name1, image_name2, label))
            line1 = pairs_file_buf.readline().strip()
        assert(len(pair_list) == 6000)
        test_pair_list = []
        positive_start = 0 # 0-2999
        negtive_start = 3000 # 3000 - 5999
        for set_idx in range(10):
            positive_index = positive_start + 300 * set_idx
            negtive_index = negtive_start + 300 * set_idx
            cur_positive_pair_list = pair_list[positive_index : positive_index + 300]
            cur_negtive_pair_list = pair_list[negtive_index : negtive_index + 300]
            test_pair_list.extend(cur_positive_pair_list)
            test_pair_list.extend(cur_negtive_pair_list)
        return test_pair_list

class AgeDB_PairsParser(PairsParser):
    """The pairs parser for agedb.
    """
    def parse_pairs(self):
        test_pair_list = []
        pairs_data = scio.loadmat(self.pairs_file)
        splits = pairs_data['splits']
        for split_index in range(10):
            cur_split = splits[split_index]
            cur_pairs = cur_split[0][0][0][0]
            cur_labels = cur_split[0][0][0][1][0]
            cur_first_list = cur_pairs[0]
            cur_second_list = cur_pairs[1]
            for pair_index in range(600):
                cur_first = cur_first_list[pair_index]
                cur_first_name = cur_first[0][0][0][0] + '.jpg'
                cur_second = cur_second_list[pair_index]
                cur_second_name = cur_second[0][0][0][0] + '.jpg'
                cur_label = cur_labels[pair_index]
                if cur_label == -1:
                    cur_label = 0
                test_pair_list.append((cur_first_name, cur_second_name, cur_label))
        return test_pair_list

class PairsParserFactory(object):
    """The factory used to produce different pairs parser for different dataset.

    Attributes:
        pairs_file(str): the path of the pairs file that was released by official.
        test_set(str): the name of different dataset.
    """
    def __init__(self, pairs_file, test_set):
        self.pairs_file = pairs_file
        self.test_set = test_set
    def get_parser(self):
        if self.test_set == 'LFW':
            pairs_parser =  LFW_PairsParser(self.pairs_file)
        elif self.test_set == 'CPLFW':
            pairs_parser = CPLFW_PairsParser(self.pairs_file)
        elif self.test_set == 'CALFW':
            pairs_parser = CALFW_PairsParser(self.pairs_file)
        elif self.test_set == 'AgeDB30':
            pairs_parser = AgeDB_PairsParser(self.pairs_file)
        elif 'RFW' in self.test_set:
            pairs_parser = RFW_PairsParser(self.pairs_file)
        else:
            pairs_parser = None
        return pairs_parser
