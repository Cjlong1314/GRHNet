import os
import torch
import numpy as np
from scipy import sparse
import scipy.sparse as sp
from config import args
from tqdm import tqdm

# todo 获取四元组列表
def load_quadruples(inPath, fileName, fileName2=None):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            rel = int(line_split[1])
            tail = int(line_split[2])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                rel = int(line_split[1])
                tail = int(line_split[2])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)
    times = list(times)
    times.sort()

    return np.asarray(quadrupleList), np.asarray(times)

# todo 获取实体数和关系数
def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])

# todo 获取指定时刻的三元组列表
def get_data_with_t(data, tim):
    triples = [[quad[0], quad[1], quad[2]] for quad in data if quad[3] == tim]
    return np.array(triples)


train_data, train_times = load_quadruples('./data/{}'.format(args.dataset), 'train.txt')
# todo 除了ICEWS14数据集，其他数据集均使用 train + valid 的数据进行历史统计
if args.dataset != 'ICEWS14':
    valid_data, valid_times = load_quadruples('./data/{}'.format(args.dataset), 'valid.txt')
    train_data = np.concatenate((train_data, valid_data), axis=0)
    train_times = np.concatenate((train_times, valid_times), axis=0)
num_e, num_r = get_total_number('./data/{}'.format(args.dataset), 'stat.txt')
save_dir_obj = './data/{}/copy_seq/'.format(args.dataset)
save_dir_sub = './data/{}/copy_seq_sub/'.format(args.dataset)


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


mkdirs(save_dir_obj)
mkdirs(save_dir_sub)

accumulated_data = {}
train_new_data = []
for tim in tqdm(train_times):
    train_new_data.append([[quad[0], quad[1], quad[2], quad[3]] for quad in train_data if quad[3] == tim])
j = 0
# todo 存储所有事实发生的频率
for tim in tqdm(train_times):
    # todo train_new_data[j]：存储第j个历史子图中的事实
    for data in train_new_data[j]:
        key = (data[0], data[1], data[2])
        if key in accumulated_data:
            accumulated_data[key] += 1 
        else:
            accumulated_data[key] = 1
    j += 1
    frequency = [v for _, v in accumulated_data.items()]
    s = [r for r, _, _ in accumulated_data.keys()]
    p = [c for _, c, _ in accumulated_data.keys()]
    o = [s for _, _, s in accumulated_data.keys()]
    i = 0
    k = [0] * len(s)
    while i < len(s):
        k[i] = s[i] * num_r + p[i]
        i += 1
    v = o
    tail_seq = sp.csr_matrix((frequency, (k, v)), shape=(num_e * num_r, num_e))
    sp.save_npz('./data/{}/copy_seq/train_h_r_copy_seq_{}.npz'.format(args.dataset, tim), tail_seq)
    i = 0
    k1 = [0] * len(s)
    while i < len(s):
        k1[i] = o[i] * num_r + p[i]
        i += 1
    v1 = s
    tail_seq_sub = sp.csr_matrix((frequency, (k1, v1)), shape=(num_e * num_r, num_e))
    sp.save_npz('./data/{}/copy_seq_sub/train_h_r_copy_seq_{}.npz'.format(args.dataset, tim), tail_seq_sub)

# todo 候选实体，即当前历史子图中的可见实体

j = 0
for tim in tqdm(train_times):
    # todo train_new_data[j]：存储第j个历史子图中的事实
    accumulated_data = {}
    for candidate_data in train_new_data[j]:
        key = (candidate_data[0], candidate_data[1], candidate_data[2])
        accumulated_data[key] = 1
    j += 1
    candidate_answer = [v for _, v in accumulated_data.items()]
    s = [r for r, _, _ in accumulated_data.keys()]
    p = [c for _, c, _ in accumulated_data.keys()]
    o = [s for _, _, s in accumulated_data.keys()]
    i = 0
    k = [0] * len(s)
    while i < len(s):
        k[i] = s[i] * num_r + p[i]
        i += 1
    v = o
    tail_seq = sp.csr_matrix((candidate_answer, (k, v)), shape=(num_e * num_r, num_e))
    sp.save_npz('./data/{}/copy_seq/train_h_r_candidate_seq_{}.npz'.format(args.dataset, tim), tail_seq)
    i = 0
    k1 = [0] * len(s)
    while i < len(s):
        k1[i] = o[i] * num_r + p[i]
        i += 1
    v1 = s
    tail_seq_sub = sp.csr_matrix((candidate_answer, (k1, v1)), shape=(num_e * num_r, num_e))
    sp.save_npz('./data/{}/copy_seq_sub/train_h_r_candidate_seq_{}.npz'.format(args.dataset, tim), tail_seq_sub)

# todo 全0弃用历史子图

j = 0
for tim in tqdm(train_times):
    accumulated_data = {}
    for zero_data in train_new_data[j]:
        key = (zero_data[0], zero_data[1], zero_data[2])
        accumulated_data[key] = 0
    j += 1
    zero_data = [v for _, v in accumulated_data.items()]
    s = [r for r, _, _ in accumulated_data.keys()]
    p = [c for _, c, _ in accumulated_data.keys()]
    o = [s for _, _, s in accumulated_data.keys()]
    i = 0
    k = [0] * len(s)
    while i < len(s):
        k[i] = s[i] * num_r + p[i]
        i += 1
    v = o
    tail_seq = sp.csr_matrix((zero_data, (k, v)), shape=(num_e * num_r, num_e))
    sp.save_npz('./data/{}/copy_seq/zero.npz'.format(args.dataset), tail_seq)
    i = 0
    k1 = [0] * len(s)
    while i < len(s):
        k1[i] = o[i] * num_r + p[i]
        i += 1
    v1 = s
    tail_seq_sub = sp.csr_matrix((zero_data, (k1, v1)), shape=(num_e * num_r, num_e))
    sp.save_npz('./data/{}/copy_seq_sub/zero.npz'.format(args.dataset, tim), tail_seq_sub)
