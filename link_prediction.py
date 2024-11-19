import math
import torch
from torch.nn import Parameter, init
from torch import nn
import torch.nn.functional as F
import numpy as np
from config import args
import scipy.sparse as sp
from torch.nn.init import xavier_normal_


class TConvE(torch.nn.Module):
    def __init__(self, args, num_entities, num_relations):
        super(TConvE, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, args.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args.input_drop)
        self.hidden_drop = torch.nn.Dropout(args.hidden_drop)
        self.feature_map_drop = torch.nn.Dropout2d(args.feat_drop)
        self.loss = torch.nn.BCELoss()
        self.emb_dim1 = args.embedding_shape1
        self.emb_dim2 = args.embedding_dim // self.emb_dim1

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=args.use_bias)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(args.embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(args.hidden_size, args.embedding_dim)
        print(num_entities, num_relations)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel, time):
        ent_embedded = e1.view(-1, 1, self.emb_dim1, self.emb_dim2)
        rel_embedded = rel.view(-1, 1, self.emb_dim1, self.emb_dim2)
        t_embedded = time.view(-1, 1, self.emb_dim1, self.emb_dim2)
        stacked_inputs = torch.cat([ent_embedded, rel_embedded, t_embedded], 2)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.emb_e.weight.transpose(1, 0))
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)
        return pred


#####################################################################################################


class link_prediction(nn.Module):
    def __init__(self, i_dim, h_dim, num_rels, num_times, use_cuda=False, dataset='YAGO'):
        super(link_prediction, self).__init__()
        self.dataset = dataset
        self.i_dim = i_dim
        self.h_dim = h_dim
        self.num_rels = num_rels
        self.num_times = num_times
        self.use_cuda = use_cuda
        self.ent_init_embeds = nn.Parameter(torch.Tensor(i_dim, h_dim))
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        self.tim_init_embeds = nn.Parameter(torch.Tensor(1, h_dim))
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.global_history = Global_history(self.h_dim, self.i_dim, use_cuda, self.num_rels)
        self.recent_history = Recent_history(self.h_dim, self.i_dim, use_cuda, self.num_rels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.ent_init_embeds, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.w_relation, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.tim_init_embeds, gain=nn.init.calculate_gain('relu'))

    def get_init_time(self, quadrupleList):
        T_idx = quadrupleList[:, 3] / args.time_stamp
        init_tim = torch.Tensor(self.num_times, self.h_dim)
        for i in range(self.num_times):
            init_tim[i] = torch.Tensor(self.tim_init_embeds.cpu().detach().numpy().reshape(self.h_dim)) * (i + 1)
        init_tim = init_tim.to("cuda:{}".format(args.gpu))
        T = init_tim[T_idx]
        return T

    # todo 新增
    def get_s_p_o(self, quadrupleList):
        s_idx = quadrupleList[:, 0]
        p_idx = quadrupleList[:, 1]
        o_idx = quadrupleList[:, 2]
        s = self.ent_init_embeds[s_idx]
        o = self.ent_init_embeds[o_idx]
        # todo  后面去对关系进行处理：生成两种不同的关系来进行操作
        p_o = self.w_relation[p_idx]  # * s + o
        p_s = self.w_relation[p_idx]  # * o + s
        return s, p_o, p_s, o

    # todo  新增
    def forward(self, quadruple, copy_vocabulary, candidate_table, entity):
        s, p_o, p_s, o = self.get_s_p_o(quadruple)
        p = [p_o, p_s]
        T = self.get_init_time(quadruple)
        # todo=================
        his_len = args.his_len
        if args.dataset in ['YAGO','WIKI','ICEWS14','ICEWS18','GDELT']:
            if his_len != 0:
                score_g = self.global_history(s, p, o, T, copy_vocabulary, entity)
                score_r = self.recent_history(s, p, o, T, candidate_table, entity)
                a = args.alpha
                score = score_r * a + score_g * (1 - a)
                # score = score_g
                # score = score_r
                score_end = torch.log(score)
            else:
                score_g = self.global_history(s, p, o, T, copy_vocabulary, entity)
                score_end = torch.log(score_g)
            return score_end

    def regularization_loss(self, reg_param):
        regularization_loss = torch.mean(self.w_relation.pow(2)) + torch.mean(self.ent_init_embeds.pow(2)) + torch.mean(
            self.tim_init_embeds.pow(2))
        return regularization_loss * reg_param


class Global_history(nn.Module):
    def __init__(self, hidden_dim, output_dim, use_cuda, num_rels):
        super(Global_history, self).__init__()
        self.hidden_dim = hidden_dim
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.W_s = nn.Linear(hidden_dim * 3, output_dim)
        self.use_cuda = use_cuda
        self.dropout = nn.Dropout(args.dropout)
        self.TConvE = TConvE(args, output_dim, num_rels)
    def forward(self, s_embed, rel_embed, o_embed, time_embed, copy_vocabulary, entity):
        if entity == 'object':
            m_t = self.TConvE(s_embed, rel_embed[0], time_embed)
        if entity == 'subject':
            m_t = self.TConvE(o_embed, rel_embed[1], time_embed)
        q_s = m_t
        encoded_mask = torch.Tensor(np.array(copy_vocabulary.cpu() != 0, dtype=float) * 1)
        encoded_mask += torch.Tensor(np.array(copy_vocabulary.cpu() == 0, dtype=float) * -50)
        if self.use_cuda:
            encoded_mask = encoded_mask.to("cuda:{}".format(args.gpu))
        score_s = q_s + encoded_mask
        return F.softmax(score_s, dim=1)


class Recent_history(nn.Module):
    def __init__(self, hidden_dim, output_dim, use_cuda, num_rels):
        super(Recent_history, self).__init__()
        self.hidden_dim = hidden_dim
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.W_s = nn.Linear(hidden_dim * 3, output_dim)
        self.use_cuda = use_cuda
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(args.dropout)
        self.TConvE = TConvE(args, output_dim, num_rels)

    def forward(self, s_embed, rel_embed, o_embed, time_embed, copy_vocabulary, entity):
        if entity == 'object':
            m_t = self.TConvE(s_embed, rel_embed[0], time_embed)
        if entity == 'subject':
            m_t = self.TConvE(o_embed, rel_embed[1], time_embed)
        q_s = m_t
        encoded_mask = torch.Tensor(np.array(copy_vocabulary.cpu() != 0, dtype=float) * 1)
        encoded_mask += torch.Tensor(np.array(copy_vocabulary.cpu() == 0, dtype=float) * -50)
        if self.use_cuda:
            encoded_mask = encoded_mask.to("cuda:{}".format(args.gpu))
        score_g = q_s + encoded_mask

        return F.softmax(score_g, dim=1)
