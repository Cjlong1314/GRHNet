import time
import os
from utils import *
import torch.nn.functional as F
from torch import nn
from torch import optim
import torch
import math
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from config import args
from link_prediction import link_prediction
from evolution import calc_raw_mrr, calc_filtered_mrr, calc_filtered_test_mrr
import codecs
import json
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

torch.set_num_threads(2)

import warnings
warnings.filterwarnings(action='ignore')


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

his_len = args.his_len
use_cuda = args.gpu >= 0 and torch.cuda.is_available()
device = torch.device("cuda:{}".format(args.gpu) if use_cuda else "cpu")

if args.dataset == 'ICEWS14':
    train_data, train_times = load_quadruples('./data/{}'.format(args.dataset), 'train.txt')
    test_data, test_times = load_quadruples('./data/{}'.format(args.dataset), 'test.txt')
    dev_data, dev_times = load_quadruples('./data/{}'.format(args.dataset), 'test.txt')
else:
    train_data, train_times = load_quadruples('./data/{}'.format(args.dataset), 'train.txt')
    test_data, test_times = load_quadruples('./data/{}'.format(args.dataset), 'test.txt')
    dev_data, dev_times = load_quadruples('./data/{}'.format(args.dataset), 'valid.txt')

all_times = np.concatenate([train_times, dev_times, test_times])

num_e, num_r = get_total_number('./data/{}'.format(args.dataset), 'stat.txt')
num_times = int(max(all_times) / args.time_stamp) + 1
print('num_times', num_times)

train_samples = []
for tim in train_times:
    print(str(tim) + '\t' + str(max(train_times)))
    data = get_data_with_t(train_data, tim)
    train_samples.append(len(data))

model = link_prediction(num_e, args.hidden_dim, num_r, num_times, use_cuda, args.dataset)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)

if args.entity == 'object':
    mkdirs('./results/bestmodel/{}'.format(args.dataset))
    model_state_file = './results/bestmodel/{}/model_state.pth'.format(args.dataset)
if args.entity == 'subject':
    mkdirs('./results/bestmodel/{}_sub'.format(args.dataset))
    model_state_file = './results/bestmodel/{}_sub/model_state.pth'.format(args.dataset)

forward_time = []
backward_time = []

best_mrr = 0
best_hits1 = 0
best_hits3 = 0
best_hits10 = 0

batch_size = args.batch_size
mkdirs('./scalar/{}/'.format(args.dataset))
writer = SummaryWriter(log_dir='scalar/{}/'.format(args.dataset))

print('start train')
n = 0

for i in range(args.n_epochs):
    print("正在训练第{}轮,训练{}".format((i + 1), args.entity))
    train_loss = 0
    sample_read = 0
    sample_end = 0

    if his_len == 0:
        # todo train_samples_tim：某个历史子图的序列号
        for train_samples_tim in range(len(train_samples)):
            model.train()
            # todo train_samples[train_samples_tim]: 存储当前历史子图的事实的数量； sample_end：存储所有历史子图的事实的数量
            sample_end += train_samples[train_samples_tim]
            # todo train_tim是当前历史子图的时间戳
            train_tim = train_times[train_samples_tim]
            if args.entity == 'object':
                tim_tail_seq = sp.load_npz(
                    './data/{}/copy_seq/train_h_r_copy_seq_{}.npz'.format(args.dataset, train_tim))
                tim_candidate_seq = sp.load_npz('./data/{}/copy_seq/zero.npz'.format(args.dataset))
            if args.entity == 'subject':
                tim_tail_seq = sp.load_npz(
                    './data/{}/copy_seq_sub/train_h_r_copy_seq_{}.npz'.format(args.dataset, train_tim))
                tim_candidate_seq = sp.load_npz('./data/{}/copy_seq_sub/zero.npz'.format(args.dataset))
            # todo train_sample_data：存储所有历史子图的事实
            train_sample_data = train_data[sample_read:sample_end, :]

            n_batch = (train_sample_data.shape[0] + batch_size - 1) // batch_size
            for idx in range(n_batch):
                batch_start = idx * batch_size
                batch_end = min(train_sample_data.shape[0], (idx + 1) * batch_size)
                train_batch_data = train_sample_data[batch_start: batch_end]

                if args.entity == 'object':
                    labels = torch.LongTensor(train_batch_data[:, 2])
                    seq_idx = train_batch_data[:, 0] * num_r + train_batch_data[:, 1]
                if args.entity == 'subject':
                    labels = torch.LongTensor(train_batch_data[:, 0])
                    seq_idx = train_batch_data[:, 2] * num_r + train_batch_data[:, 1]
                tail_seq = torch.Tensor(tim_tail_seq[seq_idx].todense())
                # todo=============
                candidate_seq = torch.Tensor(tim_candidate_seq[seq_idx].todense())
                if use_cuda:
                    # todo==========
                    labels, tail_seq, candidate_seq = labels.to(device), tail_seq.to(device), candidate_seq.to(
                        device)
                t0 = time.time()
                score = model(train_batch_data, tail_seq, candidate_seq, entity=args.entity)
                loss = F.nll_loss(score, labels) + model.regularization_loss(reg_param=0.01)
                train_loss += loss.item()
                t1 = time.time()
                loss.backward()
                optimizer.step()
                t2 = time.time()
                writer.add_scalar('{}_loss'.format(args.dataset), loss.item(), n)
                n += 1
                forward_time.append(t1 - t0)
                backward_time.append(t2 - t1)
                optimizer.zero_grad()
            sample_read += train_samples[train_samples_tim]
    else:
        for train_samples_tim in range(len(train_samples)):
            model.train()
            # todo sample_end存储当前历史子图的所有事件的数量
            sample_end += train_samples[train_samples_tim]
            # todo train_tim是当前历史子图的时间戳
            train_tim = train_times[train_samples_tim]
            # todo K个历史子图===========
            if train_tim / args.time_stamp > (his_len - 1):
                if args.entity == 'object':
                    tim_candidate_seq = sp.load_npz(
                            './data/{}/copy_seq/train_h_r_candidate_seq_{}.npz'.format(args.dataset, train_times[train_samples_tim]))
                    for m in range(his_len - 1):
                        tim_candidate_seq += sp.load_npz(
                            './data/{}/copy_seq/train_h_r_candidate_seq_{}.npz'.format(args.dataset, train_times[train_samples_tim - (m + 1)]))
                if args.entity == 'subject':
                    tim_candidate_seq = sp.load_npz(
                            './data/{}/copy_seq_sub/train_h_r_candidate_seq_{}.npz'.format(args.dataset, train_times[train_samples_tim]))
                    for n in range(his_len - 1):
                        tim_candidate_seq += sp.load_npz(
                            './data/{}/copy_seq_sub/train_h_r_candidate_seq_{}.npz'.format(args.dataset, train_times[train_samples_tim - (n + 1)]))
            else:
                if args.entity == 'object':
                    tim_candidate_seq = sp.load_npz('./data/{}/copy_seq/zero.npz'.format(args.dataset))
                if args.entity == 'subject':
                    tim_candidate_seq = sp.load_npz('./data/{}/copy_seq_sub/zero.npz'.format(args.dataset))
            if args.entity == 'object':
                tim_tail_seq = sp.load_npz(
                    './data/{}/copy_seq/train_h_r_copy_seq_{}.npz'.format(args.dataset, train_tim))
            if args.entity == 'subject':
                tim_tail_seq = sp.load_npz(
                    './data/{}/copy_seq_sub/train_h_r_copy_seq_{}.npz'.format(args.dataset, train_tim))
            train_sample_data = train_data[sample_read:sample_end, :]
            n_batch = (train_sample_data.shape[0] + batch_size - 1) // batch_size
            for idx in range(n_batch):
                batch_start = idx * batch_size
                batch_end = min(train_sample_data.shape[0], (idx + 1) * batch_size)
                train_batch_data = train_sample_data[batch_start: batch_end]
                if args.entity == 'object':
                    labels = torch.LongTensor(train_batch_data[:, 2])
                    seq_idx = train_batch_data[:, 0] * num_r + train_batch_data[:, 1]
                if args.entity == 'subject':
                    labels = torch.LongTensor(train_batch_data[:, 0])
                    seq_idx = train_batch_data[:, 2] * num_r + train_batch_data[:, 1]
                tail_seq = torch.Tensor(tim_tail_seq[seq_idx].todense())
                # todo=============
                candidate_seq = torch.Tensor(tim_candidate_seq[seq_idx].todense())
                if use_cuda:
                    # todo==========
                    labels, tail_seq, candidate_seq = labels.to(device), tail_seq.to(device), candidate_seq.to(device)
                t0 = time.time()
                score = model(train_batch_data, tail_seq, candidate_seq, entity=args.entity)
                loss = F.nll_loss(score, labels) + model.regularization_loss(reg_param=0.01)
                train_loss += loss.item()
                t1 = time.time()
                loss.backward()
                optimizer.step()
                t2 = time.time()
                writer.add_scalar('{}_loss'.format(args.dataset), loss.item(), n)
                n += 1
                forward_time.append(t1 - t0)
                backward_time.append(t2 - t1)
                optimizer.zero_grad()
            sample_read += train_samples[train_samples_tim]
    print("total_loss:{}".format(train_loss))

    if (i + 1) % args.valid_epoch == 0:
        mrr, hits1, hits3, hits10 = 0, 0, 0, 0

        if args.entity == 'object':
            tim_tail_seq = sp.load_npz(
                './data/{}/copy_seq/train_h_r_copy_seq_{}.npz'.format(args.dataset, train_times[-1]))
        if args.entity == 'subject':
            tim_tail_seq = sp.load_npz(
                './data/{}/copy_seq_sub/train_h_r_copy_seq_{}.npz'.format(args.dataset, train_times[-1]))

        # todo =======================
        if his_len != 0:
            tim_candidate_seq = []
            if args.entity == 'object':
                tim_candidate_seq = sp.load_npz(
                        './data/{}/copy_seq/train_h_r_candidate_seq_{}.npz'.format(args.dataset, train_times[-1]))
                for p in range(his_len - 1):
                    tim_candidate_seq += sp.load_npz(
                        './data/{}/copy_seq/train_h_r_candidate_seq_{}.npz'.format(args.dataset, train_times[-(2 + p)]))
            if args.entity == 'subject':
                tim_candidate_seq = sp.load_npz(
                        './data/{}/copy_seq_sub/train_h_r_candidate_seq_{}.npz'.format(args.dataset, train_times[-1]))
                for q in range(his_len - 1):
                    tim_candidate_seq += sp.load_npz(
                        './data/{}/copy_seq_sub/train_h_r_candidate_seq_{}.npz'.format(args.dataset, train_times[-(2 + q)]))
        else:
            if args.entity == 'object':
                tim_candidate_seq = sp.load_npz('./data/{}/copy_seq/zero.npz'.format(args.dataset))
            if args.entity == 'subject':
                tim_candidate_seq = sp.load_npz('./data/{}/copy_seq_sub/zero.npz'.format(args.dataset))
        # todo =======================

        n_batch = (dev_data.shape[0] + batch_size - 1) // batch_size

        for idx in range(n_batch):
            # todo 读取小样本数据进行测试
            batch_start = idx * batch_size
            batch_end = min(dev_data.shape[0], (idx + 1) * batch_size)
            dev_batch_data = dev_data[batch_start: batch_end]

            if args.entity == 'object':
                # todo 真值标签
                dev_label = torch.LongTensor(dev_batch_data[:, 2])
                # todo 序列化索引
                seq_idx = dev_batch_data[:, 0] * num_r + dev_batch_data[:, 1]
            if args.entity == 'subject':
                dev_label = torch.LongTensor(dev_batch_data[:, 0])
                seq_idx = dev_batch_data[:, 2] * num_r + dev_batch_data[:, 1]
            # todo 反序列化读取数据
            tail_seq = torch.Tensor(tim_tail_seq[seq_idx].todense())
            # todo===================
            candidate_seq = torch.Tensor(tim_candidate_seq[seq_idx].todense())

            if use_cuda:
                # todo===================
                dev_label, tail_seq, candidate_seq = dev_label.to(device), tail_seq.to(device), candidate_seq.to(device)
            dev_score = model(dev_batch_data, tail_seq, candidate_seq, entity=args.entity)

            if args.raw:
                tim_mrr, tim_hits1, tim_hits3, tim_hits10 = calc_raw_mrr(dev_score, dev_label, hits=[1, 3, 10])
            else:
                tim_mrr, tim_hits1, tim_hits3, tim_hits10 = calc_filtered_mrr(num_e,
                                                                              dev_score,
                                                                              torch.LongTensor(train_data),
                                                                              torch.LongTensor(dev_data),
                                                                              torch.LongTensor(dev_batch_data),
                                                                              entity=args.entity,
                                                                              hits=[1, 3, 10])

            mrr += tim_mrr * len(dev_batch_data)
            hits1 += tim_hits1 * len(dev_batch_data)
            hits3 += tim_hits3 * len(dev_batch_data)
            hits10 += tim_hits10 * len(dev_batch_data)

        mrr = mrr / dev_data.shape[0]
        hits1 = hits1 / dev_data.shape[0]
        hits3 = hits3 / dev_data.shape[0]
        hits10 = hits10 / dev_data.shape[0]
        print("MRR : {:.6f}".format(mrr))
        print("Hits @ 1: {:.6f}".format(hits1))
        print("Hits @ 3: {:.6f}".format(hits3))
        print("Hits @ 10: {:.6f}".format(hits10))
        if mrr > best_mrr:
            best_mrr = mrr
            torch.save({'state_dict': model.state_dict(), 'epoch': i + 1}, model_state_file)
            count = 0
        else:
            count += 1

        if hits1 > best_hits1:
            best_hits1 = hits1
        if hits3 > best_hits3:
            best_hits3 = hits3
        if hits10 > best_hits10:
            best_hits10 = hits10
        print(
            "Epoch {:04d} | Loss {:.4f} | Best MRR {:.4f} | Hits@1 {:.4f} | Hits@3 {:.4f} | Hits@10 {:.4f} | Forward {:.4f}s | Backward {:.4f}s".format(
                i + 1, train_loss, best_mrr, best_hits1, best_hits3, best_hits10, forward_time[-1],
                backward_time[-1]))

        if count == args.counts:
            break
writer.close()

print("training done")
print("Mean forward time: {:4f}s".format(np.mean(forward_time)))
print("Mean Backward time: {:4f}s".format(np.mean(backward_time)))
