# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import re
import os
import sys
import random
import string
import logging

import argparse, msgpack
import torch
import logging
import pickle
import pandas as pd
from collections import Counter

import torch
import numpy as np
from itertools import compress

# Modification: remove unused functions and imports, add a boolean parser.
# Origin: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa

# ------------------------------------------------------------------------------
# General logging utilities.
# ------------------------------------------------------------------------------


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum = self.sum + val * n
        self.count = self.count +   n
        self.avg = self.sum / self.count


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def setup_logger(name, log_file):
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d %I:%M')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    log.addHandler(fh)
    log.addHandler(ch)
    return log

def add_arguments(parser):
    # system
    parser.add_argument('--log_file', default='output.log',
                        help='path for log file.')
    parser.add_argument('--log_per_updates', type=int, default=400,
                        help='log model loss per x updates (mini-batches).')
    parser.add_argument('--data_file', default='SQuAD/data.msgpack',
                        help='path to preprocessed data file.')
    parser.add_argument('--model_dir', default='apip_models',
                        help='path to store saved models.')
    parser.add_argument('--save_last_only', action='store_true',
                        help='only save the final models.')
    parser.add_argument('--eval_per_epoch', type=int, default=1,
                        help='perform evaluation per x epochs.')
    parser.add_argument('--seed', type=int, default=937,
                        help='random seed for data shuffling, dropout, etc.')
    parser.add_argument("--cuda", type=str2bool, nargs='?',
                        const=True, default=torch.cuda.is_available(),
                        help='whether to use GPU acceleration.')
    # training
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-rs', '--resume', default='',
                        help='previous model file name (in `model_dir`). '
                             'e.g. "checkpoint_epoch_11.pt"')
    parser.add_argument('-rd', '--restore_dir', default='',
                        help='previous model file name (in `model_dir`). '
                             'e.g. "checkpoint_epoch_11.pt"')
    parser.add_argument('-ro', '--resume_options', action='store_true',
                        help='use previous model options, ignore the cli and defaults.')
    parser.add_argument('-rlr', '--reduce_lr', type=float, default=0.,
                        help='reduce initial (resumed) learning rate by this factor.')
    parser.add_argument('-op', '--optimizer', default='adamax',
                        help='supported optimizer: adamax, sgd')
    parser.add_argument('-gc', '--grad_clipping', type=float, default=20)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                        help='only applied to SGD.')
    parser.add_argument('-mm', '--momentum', type=float, default=0,
                        help='only applied to SGD.')
    parser.add_argument('-tp', '--tune_partial', type=int, default=1000,
                        help='finetune top-x embeddings.')
    parser.add_argument('--fix_embeddings', action='store_true',
                        help='if true, `tune_partial` will be ignored.')
    parser.add_argument('--rnn_padding', action='store_true',
                        help='perform rnn padding (much slower but more accurate).')
    # different modules
    parser.add_argument('--squad', default=1, type=int,help='SQuAD type: 1.0 or 2.0')
    
    parser.add_argument('--pi_inp', default='rnn_cat', help='input into latent policy =rnn_cat=, =mix=')
    parser.add_argument('--ae_restore', default='apip_models/08m06d_151643/best_model.pt', help='restore file for answer exist policy')
    parser.add_argument('--ae_archt', default='policy_con_4', help='=bili=, =policy=')
    parser.add_argument('--select_i', action='store_true',help='if true, train selection policy for scoring good on SQuAD')
    parser.add_argument('--vae', action='store_true',help='if true, use vae for gradient estimation')
    parser.add_argument('--interpret', action='store_true',help='if true, use induced interpretation values for testing')
    parser.add_argument('--rl_tuning', default='', type=str, help='=pgm=, =pg=, =sc=')
    parser.add_argument('--policy_critic', default='4_3', type=str, help='number of layers in MLP for policy and critic networks')
    parser.add_argument('--debug', action='store_true',help='if true, debug')
    parser.add_argument('--entropy_loss', default=0.0, type=float ,help='if true, use entropy loss')
    parser.add_argument('--batch_norm', action='store_true',help='if true, use BN')
    parser.add_argument('--summary', default=True, type=bool, help='if true, make summaries')
    parser.add_argument('--all_emb_tune', action='store_true',help='if true, tune all embs')
    parser.add_argument('--semisup', action='store_true',help='if true, use semi supervised learning')
    parser.add_argument('--weight_norm', action='store_true',help='if true, use WN')
    parser.add_argument('--drop_nn', action='store_true',help='if true, use dropouts')
    parser.add_argument('--critic_loss', action='store_true',
                        help='if true, use optimize for f1 prediction')
    parser.add_argument('--self_critic', action='store_true',
                        help='if true, use policy gradient for span prediction')
    parser.add_argument('--n_actions', type=int, default=0)
    parser.add_argument('--control_d', default='', help='=q_dc= or or =q_wa= or =d_qa= or =d_ca=')
    parser.add_argument('--fin_att', default='linear', help='=linear= or =param= or =pointer_s=')
    parser.add_argument('--gate', default='tanh_1', help='=tanh= or =fc=')
    parser.add_argument('--rl_start', type=float, default=float('inf'))
    parser.add_argument('--question_merge', default='self_attn')
    parser.add_argument('--beta', default='const_1')
    parser.add_argument('--alpha', default='const_1')
    parser.add_argument('--pi_q_rnn', default='', help='rnns to update =pi= or =q= or =pi_q=')
    parser.add_argument('--ce_frac', type=float, default=0.9)
    parser.add_argument('-gpp','--grad_prob_print', type=float, default=0.00001)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--ae_coeff', type=float, default=1.0)

    # model
    parser.add_argument('--doc_layers', type=int, default=5)
    parser.add_argument('--question_layers', type=int, default=5)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_features', type=int, default=4)
    parser.add_argument('--pos', type=str2bool, nargs='?', const=True, default=True,
                        help='use pos tags as a feature.')
    parser.add_argument('--pos_size', type=int, default=56,
                        help='how many kinds of POS tags.')
    parser.add_argument('--pos_dim', type=int, default=56,
                        help='the embedding dimension for POS tags.')
    parser.add_argument('--ner', type=str2bool, nargs='?', const=True, default=True,
                        help='use named entity tags as a feature.')
    parser.add_argument('--ner_size', type=int, default=19,
                        help='how many kinds of named entity tags.')
    parser.add_argument('--ner_dim', type=int, default=19,
                        help='the embedding dimension for named entity tags.')
    parser.add_argument('--use_qemb', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--concat_rnn_layers', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--dropout_emb', type=float, default=0.5)
    parser.add_argument('--dropout_rnn', type=float, default=0.2)
    parser.add_argument('--dropout_rnn_output', type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument('--max_len', type=int, default=15)
    parser.add_argument('--rnn_type', default='sru',
                        help='supported types: rnn, gru, lstm')
    return parser


def lr_decay(optimizer, lr_decay, log):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    log.info('[learning rate reduced by {}]'.format(lr_decay))
    return optimizer

def select_scope_update(args, epoch):
    # use VAE framework until time rl_start and the switch to RL framework
    if args.rl_start > epoch:
        scope = 'pi_q'
    elif args.rl_start <= epoch:
        if args.rl_start == epoch:
            print("\nSTARTED RL UPDATES\n")
        scope = "rl"
    return scope

def load_data(opt, args):
    # return train and development(test) sets
    # max q len = 60
    # max c len = 767
    if opt['squad'] == 1:
        squad_dir = 'SQuAD'
    else:
        squad_dir = 'SQuAD2'

    with open(os.path.join(squad_dir, 'meta.msgpack'), 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')
    embedding = torch.Tensor(meta['embedding'])
    opt['pretrained_words'] = True
    opt['vocab_size'] = embedding.size(0)
    opt['embedding_dim'] = embedding.size(1)
    if not opt['fix_embeddings']:
        embedding[1] = torch.normal(means=torch.zeros(opt['embedding_dim']), std=1.)
    with open(args.data_file, 'rb') as f:
        data = msgpack.load(f, encoding='utf8')
    if args.semisup:
        with open(os.path.join(squad_dir, 'q_labels_sm5.pickle'), 'rb') as f:
            q_labels = pickle.load(f, encoding='utf8')
            print("loading question labels for %d actions"%args.n_actions)
            q_l, ql_mask = q_labels[args.n_actions]
    else:
        q_l, ql_mask = [0]*len(data['trn_question_ids']), [0]*len(data['trn_question_ids'])
    train_orig = pd.read_csv(os.path.join(squad_dir, 'train.csv'))
    train = list(zip(
        data['trn_context_ids'],
        data['trn_context_features'],
        data['trn_context_tags'],
        data['trn_context_ents'],
        data['trn_question_ids'],
        train_orig['answer_start_token'].tolist(),
        train_orig['answer_end_token'].tolist(),
        data['trn_ans_exists'],
        data['trn_context_text'],
        data['trn_context_spans']
    ))
    train_y = train_orig['answer'].tolist()[:len(train)]
    train_y = [[y] for y in train_y]
    dev = list(zip(
        data['dev_context_ids'],
        data['dev_context_features'],
        data['dev_context_tags'],
        data['dev_context_ents'],
        data['dev_question_ids'],
        data['dev_ans_exists'],
        data['dev_context_text'],
        data['dev_context_spans']
    ))
    if not 'data2' in args.data_file and not 'data_a' in args.data_file:
        dev_orig = pd.read_csv(os.path.join(squad_dir, 'dev.csv'))
        dev_y = dev_orig['answers'].tolist()[:len(dev)]
        dev_y = [eval(y) for y in dev_y]
    else:
        dev_y = data['dev_answers']
    return train, dev, dev_y, train_y, embedding, opt, q_l, ql_mask


def load_data_train(opt, args):
    # return train and validation sets
    # max q len = 60
    # max c len = 767
    if opt['squad'] == 1:
        squad_dir = 'SQuAD'
    else:
        squad_dir = 'SQuAD2'

    with open(os.path.join(squad_dir, 'meta.msgpack'), 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')
    embedding = torch.Tensor(meta['embedding'])
    opt['pretrained_words'] = True
    opt['vocab_size'] = embedding.size(0)
    opt['embedding_dim'] = embedding.size(1)
    if not opt['fix_embeddings']:
        embedding[1] = torch.normal(means=torch.zeros(opt['embedding_dim']), std=1.)
    with open(args.data_file, 'rb') as f:
        data = msgpack.load(f, encoding='utf8')
    if args.semisup:
        with open(os.path.join(squad_dir, 'q_labels_sm5.pickle'), 'rb') as f:
            q_labels = pickle.load(f, encoding='utf8')
            print("loading question labels for %d actions"%args.n_actions)
            q_l, ql_mask = q_labels[args.n_actions]
    else:
        q_l, ql_mask = [0]*len(data['trn_question_ids']), [0]*len(data['trn_question_ids'])
    train_orig = pd.read_csv(os.path.join(squad_dir, 'train.csv'))
    train = list(zip(
        data['trn_context_ids'],
        data['trn_context_features'],
        data['trn_context_tags'],
        data['trn_context_ents'],
        data['trn_question_ids'],
        train_orig['answer_start_token'].tolist(),
        train_orig['answer_end_token'].tolist(),
        data['trn_ans_exists'],
        data['trn_context_text'],
        data['trn_context_spans']
    ))
    train_y = train_orig['answer'].tolist()[:len(train)]
    train_y = [[y] for y in train_y]
    dev = list(zip(
        data['val_context_ids'],
        data['val_context_features'],
        data['val_context_tags'],
        data['val_context_ents'],
        data['val_question_ids'],
        data['val_ans_exists'],
        data['val_context_text'],
        data['val_context_spans']
    ))
    if not 'data2' in args.data_file and not 'data_a' in args.data_file:
        dev_orig = pd.read_csv(os.path.join(squad_dir, 'valid.csv'))
        dev_y = dev_orig['answers'].tolist()[:len(dev)]
        dev_y = [eval(y) for y in dev_y]
    else:
        dev_y = data['dev_answers']
    return train, dev, dev_y, train_y, embedding, opt, q_l, ql_mask


class BatchGen:
    def __init__(self, data, batch_size, gpu, evaluation=False, shuffle=False):
        '''
        input:
            data - list of lists
            batch_size - int
        '''
        self.batch_size = batch_size
        self.eval = evaluation
        self.gpu = gpu

        # shuffle
        if not evaluation or shuffle:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
            self.indices = [indices[i:i + batch_size] for i in range(0, len(data), batch_size)]
        else:
            indices = list(range(len(data)))
            self.indices = [indices[i:i + batch_size] for i in range(0, len(data), batch_size)]
        
        # chunk into batches
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for batch in self.data:
            batch_size = len(batch)
            batch = list(zip(*batch))
            if self.eval:
                #assert len(batch) == 7
                pass
            else:
                assert len(batch) == 10

            context_len = max(len(x) for x in batch[0])
            context_id = torch.LongTensor(batch_size, context_len).fill_(0)
            for i, doc in enumerate(batch[0]):
                context_id[i, :len(doc)] = torch.LongTensor(doc)

            feature_len = len(batch[1][0][0])
            context_feature = torch.Tensor(batch_size, context_len, feature_len).fill_(0)
            for i, doc in enumerate(batch[1]):
                for j, feature in enumerate(doc):
                    context_feature[i, j, :] = torch.Tensor(feature)

            context_tag = torch.LongTensor(batch_size, context_len).fill_(0)
            for i, doc in enumerate(batch[2]):
                context_tag[i, :len(doc)] = torch.LongTensor(doc)

            context_ent = torch.LongTensor(batch_size, context_len).fill_(0)
            for i, doc in enumerate(batch[3]):
                context_ent[i, :len(doc)] = torch.LongTensor(doc)
            question_len = max(len(x) for x in batch[4])
            question_id = torch.LongTensor(batch_size, question_len).fill_(0)
            for i, doc in enumerate(batch[4]):
                question_id[i, :len(doc)] = torch.LongTensor(doc)

            context_mask = torch.eq(context_id, 0)
            question_mask = torch.eq(question_id, 0)
            if not self.eval:
                y_s = torch.LongTensor(batch[5])
                y_e = torch.LongTensor(batch[6])

            exists = torch.FloatTensor(batch[-3])
            text = list(batch[-2])
            span = list(batch[-1])
            if self.gpu:
                context_id = context_id.pin_memory()
                context_feature = context_feature.pin_memory()
                context_tag = context_tag.pin_memory()
                context_ent = context_ent.pin_memory()
                context_mask = context_mask.pin_memory()
                question_id = question_id.pin_memory()
                question_mask = question_mask.pin_memory()
            if self.eval:
                yield (context_id, context_feature, context_tag, context_ent, context_mask,
                       question_id, question_mask, exists, text, span)
            else:
                yield (context_id, context_feature, context_tag, context_ent, context_mask,
                       question_id, question_mask, y_s, y_e, exists, text, span)


def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _exact_match(pred, answers):
    if pred is None or answers is None:
        return False
    pred = _normalize_answer(pred)
    if pred == "" and (len(set(answers))==0 or len(set(answers[0]))==0 or answers[0]=='nan'):
        return True
    for a in answers:
        if pred == _normalize_answer(a):
            return True
    return False


def _f1_score(pred, answers):
    def _score(p_tokens, a_tokens):
        common = Counter(p_tokens) & Counter(a_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1. * num_same / len(p_tokens)
        recall = 1. * num_same / len(a_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    if pred is None or answers is None:
        return 0
    if len(set(answers))==0 or len(set(answers[0]))==0 or answers[0]=='nan':
        if pred == "":
            return 1
        else:
            return 0
    p_tokens = _normalize_answer(pred).split()
    scores = [_score(p_tokens, _normalize_answer(a).split()) for a in answers]
    return max(scores)


def _overlap_score(pred, answers):
    def _score(p_tokens, a_tokens):
        common = Counter(p_tokens) & Counter(a_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        recall = 1. * num_same / len(a_tokens)
        return recall

    if pred is None or answers is None:
        return 0
    if len(set(answers))==0 or len(set(answers[0]))==0 or answers[0]=='nan':
        if pred == "":
            return 1
        else:
            return 0
    p_tokens = _normalize_answer(pred).split()
    scores = [_score(p_tokens, _normalize_answer(a).split()) for a in answers]
    return max(scores)


def overlap(pred, truth):
    assert len(pred) == len(truth)
    ov = 0
    for p, t in zip(pred, truth):
        ov += _overlap_score(p, t)
    return ov


def score_test_alli(pred, truth):
    assert len(pred) == len(truth), "pred = %d, truth = %d"%(len(pred), len(truth))
    f1 = em = total = 0
    for p, t in zip(pred, truth):
        em += _exact_match(p, t)
        f1 += _f1_score(p, t)
    return em, f1

def score(pred, truth):
    assert len(pred) == len(truth), "pred = %d, truth = %d"%(len(pred), len(truth))
    f1 = em = total = 0
    for p, t in zip(pred, truth):
        total += 1
        em += _exact_match(p, t)
        f1 += _f1_score(p, t)
    em = 100. * em / total
    f1 = 100. * f1 / total
    return em, f1

def score_list(pred, truth, mask):
    assert len(pred) == len(truth), "pred = %d, truth = %d"%(len(pred), len(truth))
    f1 = []; em = []; 
    for p, t in zip(pred, truth):
        em += [_exact_match(p, t)]
        f1 += [_f1_score(p, t)]
    print(sum(1-np.array(mask)), (np.array(f1).squeeze()==1).sum(), ((np.array(mask).squeeze()==0)*(np.array(f1).squeeze()==1)).sum(),\
                                                                    ((np.array(mask).squeeze()==0)*(np.array(em).squeeze()==1)).sum())
    em = 100. * (np.array(mask).squeeze()*np.array(em)).sum() / sum(mask)
    f1 = 100. * (np.array(mask).squeeze()*np.array(f1)).sum() / sum(mask)
    return em, f1

def score_sc(pred_s, pred_m, truth):
    f1_s, f1_m = [], []
    if pred_s:
        assert len(pred_s) == len(truth)
        for ps, pm, t in zip(pred_s, pred_m, truth):
            f1_s += [_f1_score(ps, t)]
            f1_m += [_f1_score(pm, t)]
    else:
        for pm, t in zip(pred_m, truth):
            f1_m += [_f1_score(pm, t)]
    return np.array(f1_s), np.array(f1_m)

def score_em(pred_s, pred_m, truth):
    f1_s, f1_m = [], []
    if pred_s:
        assert len(pred_s) == len(truth)
        for ps, pm, t in zip(pred_s, pred_m, truth):
            f1_s += [_exact_match(ps, t)]
            f1_m += [_exact_match(pm, t)]
    else:
        for pm, t in zip(pred_m, truth):
            f1_m += [_exact_match(pm, t)]
    return np.array(f1_s), np.array(f1_m).astype(np.float32)

def load_data_ae_1(opt, args):
    # max q len = 60
    # max c len = 767
    squad_dir = 'SQuAD2'

    with open(os.path.join(squad_dir, 'meta.msgpack'), 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')
    embedding = torch.Tensor(meta['embedding'])
    opt['pretrained_words'] = True
    opt['vocab_size'] = embedding.size(0)
    opt['embedding_dim'] = embedding.size(1)
    if not opt['fix_embeddings']:
        embedding[1] = torch.normal(means=torch.zeros(opt['embedding_dim']), std=1.)
    with open('SQuAD2/data.msgpack', 'rb') as f:
        data = msgpack.load(f, encoding='utf8')
    if args.semisup:
        with open(os.path.join(squad_dir, 'q_labels_sm5.pickle'), 'rb') as f:
            q_labels = pickle.load(f, encoding='utf8')
            print("loading question labels for %d actions"%args.n_actions)
            q_l, ql_mask = q_labels[args.n_actions]
    else:
        q_l, ql_mask = [0]*len(data['trn_question_ids']), [0]*len(data['trn_question_ids'])
    train_orig = pd.read_csv(os.path.join(squad_dir, 'train.csv'))

    def filter_list(x, mask):
        return list(compress(x, mask))

    m1 = data['trn_ans_exists'].copy()
    m2 = data['dev_ans_exists'].copy()

    data['trn_context_ids'] = filter_list(data['trn_context_ids'], m1)
    data['trn_context_features'] = filter_list(data['trn_context_features'], m1)
    data['trn_context_tags'] = filter_list(data['trn_context_tags'], m1)
    data['trn_context_ents'] = filter_list(data['trn_context_ents'], m1)
    data['trn_question_ids'] = filter_list(data['trn_question_ids'], m1)
    data['trn_context_text'] = filter_list(data['trn_context_text'], m1)
    data['trn_context_spans'] = filter_list(data['trn_context_spans'], m1)
    data['trn_ans_exists'] = [1]*sum(m1)
    answer_start_token = filter_list(train_orig['answer_start_token'].tolist(), m1)
    answer_end_token = filter_list(train_orig['answer_end_token'].tolist(), m1)
    q_l = filter_list(q_l, m1)
    ql_mask = filter_list(ql_mask, m1)


    data['dev_context_ids'] = filter_list(data['dev_context_ids'], m2)
    data['dev_context_features'] = filter_list(data['dev_context_features'], m2)
    data['dev_context_tags'] = filter_list(data['dev_context_tags'], m2)
    data['dev_context_ents'] = filter_list(data['dev_context_ents'], m2)
    data['dev_question_ids'] = filter_list(data['dev_question_ids'], m2)
    data['dev_context_text'] = filter_list(data['dev_context_text'], m2)
    data['dev_context_spans'] = filter_list(data['dev_context_spans'], m2)
    data['dev_ans_exists'] = [1]*sum(m2)

    train = list(zip(
        data['trn_context_ids'],
        data['trn_context_features'],
        data['trn_context_tags'],
        data['trn_context_ents'],
        data['trn_question_ids'],
        answer_start_token,
        answer_end_token,
        data['trn_ans_exists'],
        data['trn_context_text'],
        data['trn_context_spans']
    ))
    train_y = train_orig['answer'].tolist()[:len(m1)]
    train_y = filter_list(train_y, m1)
    train_y = [[y] for y in train_y]
    dev = list(zip(
        data['dev_context_ids'],
        data['dev_context_features'],
        data['dev_context_tags'],
        data['dev_context_ents'],
        data['dev_question_ids'],
        data['dev_ans_exists'],
        data['dev_context_text'],
        data['dev_context_spans']
    ))
    if not 'data2' in args.data_file:
        dev_orig = pd.read_csv(os.path.join(squad_dir, 'dev.csv'))
        dev_y = dev_orig['answers'].tolist()[:len(m2)]
        dev_y = [eval(y) for y in dev_y]
    else:
        dev_y = data['dev_answers']
    dev_y = filter_list(dev_y, m2)

    print("original size = %d, filtered size = %d, %d"%(len(m1), len(answer_end_token), sum(m1)))
    print("original size = %d, filtered size = %d, %d"%(len(m2), len(dev_y), sum(m2)))

    return train, dev, dev_y, train_y, embedding, opt, q_l, ql_mask
