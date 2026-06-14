import random
import torch
import os
import time

import numpy as np
import pprint as pprint
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import logging
from logging.config import dictConfig
import warnings

_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_gpu(args):
    gpu_list = [int(x) for x in args.gpu.split(',')]
    print('use gpu:', gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print('create folder:', path)
        os.makedirs(path)

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


def count_acc_topk(x,y,k=5):
    _,maxk = torch.topk(x,k,dim=-1)
    total = y.size(0)
    test_labels = y.view(-1,1) 
    #top1=(test_labels == maxk[:,0:1]).sum().item()
    topk=(test_labels == maxk).sum().item()
    return float(topk/total)

def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


def save_list_to_txt(name, input_list):
    f = open(name, mode='w')
    for item in input_list:
        f.write(str(item) + '\n')
    f.close()


def confmatrix(logits,label,filename):
    
    font={'family':'DejaVu Sans','size':18}
    matplotlib.rc('font',**font)
    matplotlib.rcParams.update({'font.family':'DejaVu Sans','font.size':18})
    plt.rcParams["font.family"]="DejaVu Sans"

    pred = torch.argmax(logits, dim=1)
    cm=confusion_matrix(label, pred,normalize='true')
    #print(cm)
    clss=len(cm)
    fig = plt.figure() 
    ax = fig.add_subplot(111) 
    cax = ax.imshow(cm,cmap=plt.cm.jet) 
    if clss<=100:
        plt.yticks([0,19,39,59,79,99],[0,20,40,60,80,100],fontsize=8)
        plt.xticks([0,19,39,59,79,99],[0,20,40,60,80,100],fontsize=8)
    elif clss<=200:
        plt.yticks([0,39,79,119,159,199],[0,40,80,120,160,200],fontsize=8)
        plt.xticks([0,39,79,119,159,199],[0,40,80,120,160,200],fontsize=8)
    else:
        plt.yticks([0,199,399,599,799,999],[0,200,400,600,800,1000],fontsize=8)
        plt.xticks([0,199,399,599,799,999],[0,200,400,600,800,1000],fontsize=8)

    plt.xlabel('Predicted Label',fontsize=10)
    plt.ylabel('True Label',fontsize=10)
    plt.tight_layout()
    plt.savefig(filename+'.pdf',bbox_inches='tight')
    plt.close()

    fig = plt.figure() 
    ax = fig.add_subplot(111) 
    cax = ax.imshow(cm, cmap=plt.cm.jet, vmin=0, vmax=1.0)
    cbar = plt.colorbar(cax) # This line includes the color bar
    cbar.ax.tick_params(labelsize=8)
    if clss<=100:
        plt.yticks([0,19,39,59,79,99],[0,20,40,60,80,100],fontsize=8)
        plt.xticks([0,19,39,59,79,99],[0,20,40,60,80,100],fontsize=8)
    elif clss<=200:
        plt.yticks([0,39,79,119,159,199],[0,40,80,120,160,200],fontsize=8)
        plt.xticks([0,39,79,119,159,199],[0,40,80,120,160,200],fontsize=8)
    else:
        plt.yticks([0,199,399,599,799,999],[0,200,400,600,800,1000],fontsize=8)
        plt.xticks([0,199,399,599,799,999],[0,200,400,600,800,1000],fontsize=8)
    plt.xlabel('Predicted Label',fontsize=10)
    plt.ylabel('True Label',fontsize=10)
    plt.tight_layout()
    plt.savefig(filename+'_cbar.pdf',bbox_inches='tight')
    plt.close()

    return cm

def generalised_avg_acc(alpha, acc):
    """
    param
    alpha: the range of alpha, e.g. [0, 1, 2, 3, ..., 12]
    acc: all accuracies of each tasks, e.g [[80], [70, 65], [65, 60, 55]]..., element at position (i,j)
        indicates the accuracy of j-th task using model after trained on i-th task.
    """
    k = [len(i) - 1 for i in acc]
    res_matrix = []
    all_gen_avg_acc_different_alpha = []
    
    for a in alpha:
        res_matrix.append([])
    
        for i in range(len(acc)):
            acc0 = acc[i][0]
    
            if k[i] > 0:
                acc_new = acc[i][1:]
                g_avg_acc = (a * acc0 + sum(acc_new)) / (k[i] + a)
            else:
                g_avg_acc = acc0 if a != 0 else 0
            res_matrix[-1].append(g_avg_acc)
    
        all_gen_avg_acc = sum(res_matrix[-1]) / len(acc)
        all_gen_avg_acc_different_alpha.append(all_gen_avg_acc)
    
    return all_gen_avg_acc_different_alpha


def get_gacc(ratio, all_acc):
    alpha = [i for i in range(ratio + 1)]
    g_acc = generalised_avg_acc(alpha, all_acc)
    area = np.trapz(g_acc, x=alpha) / ((alpha[-1] - alpha[0]) * 100)
    
    return g_acc, area


import numpy as np

def compute_gacc_session_style(all_acc, base_num, incr_num, alpha_points=11):
    """
    all_acc:
      - session 0: [avg0]  (no seen/unseen split)
      - session i>=1: [seen_i, unseen_i, avg_i]
    base_num: number of base classes
    incr_num: number of incremental classes introduced per incremental session (must be >0)
    alpha_points: number of points in alpha linspace over [0,1]

    Returns:
      alpha: np.ndarray shape [alpha_points], α ∈ [0,1]
      gacc_per_session: list of lists, each entry is gAcc(α) for that session i
      auc_per_session: list of floats, AUC for each session i (normalized by 100)
    """
    assert incr_num > 0, "incr_num must be > 0 to define base_ratio."
    base_ratio = base_num / float(incr_num)  # scaling of base weight vs one incremental session
    alpha = np.linspace(0.0, 1.0, alpha_points)

    gacc_per_session = []
    auc_per_session = []

    # iterate sessions
    for i, acc_i in enumerate(all_acc):
        if i == 0:
            # session 0: no unseen; define gAcc_i(α) = acc_base (constant) for completeness
            acc_base_only = float(acc_i[0])  # avg0 == base acc at session0
            gacc_i = np.full_like(alpha, acc_base_only, dtype=float)
            auc_i = np.trapz(gacc_i, x=alpha) / 100.0
            gacc_per_session.append(gacc_i.tolist())
            auc_per_session.append(float(auc_i))
            continue

        # i >= 1: expect [seen_i, unseen_i, avg_i]
        if len(acc_i) < 2:
            raise ValueError(f"Session {i} needs seen/unseen/avg; got: {acc_i}")

        acc_b = float(acc_i[0])   # seen_i
        acc_u = float(acc_i[1])   # unseen_i

        # number of effective incremental sessions up to i (excluding base session 0)
        num_incr_sessions_eff = i  # sessions 1..i

        # Approximate Σ_j A_i^j as unseen_i * num_incr_sessions_eff (no cohort decomposition).
        sum_Aji = acc_u * num_incr_sessions_eff

        # gAcc_i(α) = (α*base_ratio*acc_b + Σ_j A_i^j) / (α*base_ratio + num_incr_sessions_eff)
        gacc_i = []
        for a in alpha:
            numerator   = a * base_ratio * acc_b + sum_Aji
            denominator = a * base_ratio + num_incr_sessions_eff
            gacc_val = numerator / denominator if denominator > 0 else acc_b
            gacc_i.append(gacc_val)

        # AUC over α∈[0,1], normalize to [0,1] by dividing by 100
        auc_i = np.trapz(gacc_i, x=alpha) / 100.0

        gacc_per_session.append(gacc_i)
        auc_per_session.append(float(auc_i))

    return alpha.tolist(), gacc_per_session, auc_per_session