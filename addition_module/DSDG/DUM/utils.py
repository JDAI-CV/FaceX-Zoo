import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import sklearn
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import pdb


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_threshold(score_file):
    with open(score_file, 'r') as file:
        lines = file.readlines()

    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        angle = float(tokens[0])
        # pdb.set_trace()
        type = int(tokens[1])
        data.append({'map_score': angle, 'label': type})
        if type == 1:
            num_real += 1
        else:
            num_fake += 1

    min_error = count  # account ACER (or ACC)
    min_threshold = 0.0
    min_ACC = 0.0
    min_ACER = 0.0
    min_APCER = 0.0
    min_BPCER = 0.0

    for d in data:
        threshold = d['map_score']

        type1 = len([s for s in data if s['map_score'] <= threshold and s['label'] == 1])
        type2 = len([s for s in data if s['map_score'] > threshold and s['label'] == 0])

        ACC = 1 - (type1 + type2) / count
        APCER = type2 / num_fake
        BPCER = type1 / num_real
        ACER = (APCER + BPCER) / 2.0

        if ACER < min_error:
            min_error = ACER
            min_threshold = threshold
            min_ACC = ACC
            min_ACER = ACER
            min_APCER = APCER
            min_BPCER = min_BPCER

    # print(min_error, min_threshold)
    return min_threshold, min_ACC, min_APCER, min_BPCER, min_ACER


def test_threshold_based(threshold, score_file):
    with open(score_file, 'r') as file:
        lines = file.readlines()

    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        angle = float(tokens[0])
        type = int(tokens[1])
        data.append({'map_score': angle, 'label': type})
        if type == 1:
            num_real += 1
        else:
            num_fake += 1

    type1 = len([s for s in data if s['map_score'] <= threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > threshold and s['label'] == 0])

    ACC = 1 - (type1 + type2) / count
    APCER = type2 / num_fake
    BPCER = type1 / num_real
    ACER = (APCER + BPCER) / 2.0

    return ACC, APCER, BPCER, ACER


def get_err_threhold(fpr, tpr, threshold):
    RightIndex = (tpr + (1 - fpr) - 1)
    right_index = np.argmax(RightIndex)
    best_th = threshold[right_index]
    err = fpr[right_index]

    differ_tpr_fpr_1 = tpr + fpr - 1.0

    right_index = np.argmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    err = fpr[right_index]

    # print(err, best_th)
    return err, best_th


# def performances(dev_scores, dev_labels, test_scores, test_labels):
def performances(map_score_val_filename, map_score_test_filename):
    # val
    with open(map_score_val_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])  # label = int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    fpr, tpr, threshold = roc_curve(val_labels, val_scores, pos_label=1)
    val_err, val_threshold = get_err_threhold(fpr, tpr, threshold)

    type1 = len([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])

    val_ACC = 1 - (type1 + type2) / count
    val_APCER = type2 / num_fake
    val_BPCER = type1 / num_real
    val_ACER = (val_APCER + val_BPCER) / 2.0

    # test 
    with open(map_score_test_filename, 'r') as file2:
        lines = file2.readlines()
    test_scores = []
    test_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])  # label = int(tokens[1])
        test_scores.append(score)
        test_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    # test based on val_threshold     
    type1 = len([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    print([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])
    print([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])

    test_ACC = 1 - (type1 + type2) / count
    test_APCER = type2 / num_fake
    test_BPCER = type1 / num_real
    test_ACER = (test_APCER + test_BPCER) / 2.0

    # test based on test_threshold     
    fpr_test, tpr_test, threshold_test = roc_curve(test_labels, test_scores, pos_label=1)
    err_test, best_test_threshold = get_err_threhold(fpr_test, tpr_test, threshold_test)

    type1 = len([s for s in data if s['map_score'] <= best_test_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > best_test_threshold and s['label'] == 0])

    test_threshold_ACC = 1 - (type1 + type2) / count
    test_threshold_APCER = type2 / num_fake
    test_threshold_BPCER = type1 / num_real
    test_threshold_ACER = (test_threshold_APCER + test_threshold_BPCER) / 2.0

    return val_threshold, best_test_threshold, val_ACC, val_ACER, test_ACC, test_APCER, test_BPCER, test_ACER, test_threshold_ACER


def performances_SiW_EER(map_score_val_filename):
    # val
    with open(map_score_val_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    fpr, tpr, threshold = roc_curve(val_labels, val_scores, pos_label=1)
    val_err, val_threshold = get_err_threhold(fpr, tpr, threshold)

    type1 = len([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])

    val_ACC = 1 - (type1 + type2) / count
    val_APCER = type2 / num_fake
    val_BPCER = type1 / num_real
    val_ACER = (val_APCER + val_BPCER) / 2.0

    return val_threshold, val_ACC, val_APCER, val_BPCER, val_ACER


def performances_SiWM_EER(map_score_val_filename):
    # val
    with open(map_score_val_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    fpr, tpr, threshold = roc_curve(val_labels, val_scores, pos_label=1)
    val_err, val_threshold = get_err_threhold(fpr, tpr, threshold)

    type1 = len([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])

    val_ACC = 1 - (type1 + type2) / count
    val_APCER = type2 / num_fake
    val_BPCER = type1 / num_real
    val_ACER = (val_APCER + val_BPCER) / 2.0

    return val_threshold, val_err, val_ACC, val_APCER, val_BPCER, val_ACER


def get_err_threhold_CASIA_Replay(fpr, tpr, threshold):
    RightIndex = (tpr + (1 - fpr) - 1)
    right_index = np.argmax(RightIndex)
    best_th = threshold[right_index]
    err = fpr[right_index]

    differ_tpr_fpr_1 = tpr + fpr - 1.0

    right_index = np.argmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    err = fpr[right_index]

    # print(err, best_th)
    return err, best_th, right_index


def performances_CASIA_Replay(map_score_val_filename):
    # val
    with open(map_score_val_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])  # int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    fpr, tpr, threshold = roc_curve(val_labels, val_scores, pos_label=1)
    val_err, val_threshold, right_index = get_err_threhold_CASIA_Replay(fpr, tpr, threshold)

    type1 = len([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    print([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])
    print([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])

    val_ACC = 1 - (type1 + type2) / count

    FRR = 1 - tpr  # FRR = 1 - TPR

    HTER = (fpr + FRR) / 2.0  # error recognition rate &  reject recognition rate

    return val_ACC, fpr[right_index], FRR[right_index], HTER[right_index], val_threshold


def performances_ZeroShot(map_score_val_filename):
    # val
    with open(map_score_val_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    fpr, tpr, threshold = roc_curve(val_labels, val_scores, pos_label=1)
    auc_val = metrics.auc(fpr, tpr)

    val_err, val_threshold, right_index = get_err_threhold_CASIA_Replay(fpr, tpr, threshold)

    type1 = len([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])

    val_ACC = 1 - (type1 + type2) / count

    FRR = 1 - tpr  # FRR = 1 - TPR

    HTER = (fpr + FRR) / 2.0  # error recognition rate &  reject recognition rate

    return val_ACC, auc_val, HTER[right_index]


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
        dst_file = os.path.join(path, 'scripts', os.path.basename(script))
        shutil.copyfile(script, dst_file)
