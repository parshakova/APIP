# get scores for F1 Threshold(rho) experiments for APIP flavors

import re
import os
import sys
import random
import argparse
import json
from datetime import datetime
from collections import OrderedDict

import msgpack, time
from tqdm import tqdm
import numpy as np

import torch
from torch.autograd import Variable

from apip import utils
from apip.model import DocReaderModel

parser = argparse.ArgumentParser(
    description='Train a Document Reader model.'
)
parser = utils.add_arguments(parser)
args = parser.parse_args()
if not args.drop_nn:
    args.dropout_rate = 0.

# set model dir
model_dir = args.model_dir
model_dir = os.path.abspath(model_dir)
torch.set_printoptions(precision=10)
# save model configuration
s = "\nParameters:\n"
for k in sorted(args.__dict__):
    s += "{} = {} \n".format(k, args.__dict__[k])
print(s)
# set random seed
seed = args.seed if args.seed >= 0 else int(random.random()*1000)
print ('seed:', seed)
random.seed(seed)
torch.manual_seed(seed)
if args.cuda:
    torch.cuda.manual_seed(seed)

def accuracies_on_ds(data_file, inputs, model, n_ans):
    train, dev, dev_y, train_y, embedding, opt, q_labels, ql_mask = inputs

    model.opt['interpret'] = False
    batches = utils.BatchGen(dev, batch_size=args.batch_size, evaluation=True, gpu=args.cuda)
    predictions = []
    pred_answers = {}
    for i, batch in enumerate(batches):
        pred = model.predict(batch)[0]
        predictions.extend(pred)

    em, f1 = utils.score(predictions, dev_y)

    print("[EM: {0:.2f} F1: {1:.2f}] on {2}".format(em, f1, data_file))

    batches = utils.BatchGen(dev, batch_size=args.batch_size, evaluation=True, gpu=args.cuda, shuffle=True)
    model.opt['interpret'] = True
    t_a, t_total_a = {0.1:0, 0.2:0, 0.3:0, 0.4:0, 0.5:0, 0.6:0, 0.7:0, 0.8:0, 0.9:0}, 0
    f1s_a = []; ovs_a = []
    # evaluate the model for all interpretations and all answers
    # if f1 score for all GT answers is > p then count answer as correct
    for i, batch in tqdm(enumerate(batches)):
        i_predictions = []
        truth = np.take(dev_y, batches.indices[i], 0)
        if args.n_actions>0:
            for a in range(args.n_actions):
                latent_a = Variable(torch.ones(batch[0].size(0))*a).long().cuda()
                pred = model.predict_inter(batch, latent_a=latent_a)
                i_predictions.append(pred[0])
        else:
            i_predictions = model.predict(batch)[0]
        for b in range(batch[0].size(0)):
            f1s = []
            for ta in truth[b]:
                f1_v = []
                for a in range(args.n_actions):
                    _, f1_a = utils.score_test_alli([i_predictions[a][b]], [[ta]])
                    f1_v += [f1_a]
                if args.n_actions>0:
                    f1s += [max(f1_v)]
                else:
                    _, f1_v = utils.score_test_alli([i_predictions[b]], [[ta]])
                    f1s += [f1_v]
            f1s = np.array(f1s)
            for p in t_a.keys():
                t_a[p] = t_a[p] + int((f1s>p).sum() == n_ans)

            f1_i = []; ov_i = []
            for a in range(args.n_actions):
                _, f1_a = utils.score_test_alli([i_predictions[a][b]], [truth[b]])
                ov_a = utils.overlap([i_predictions[a][b]], [truth[b]])
                f1_i += [f1_a]; ov_i += [ov_a]
            
            if args.n_actions == 0:
                _, f1_i = utils.score_test_alli([i_predictions[b]], [truth[b]])
                ov_i = utils.overlap([i_predictions[b]], [truth[b]])
            f1s_a += [f1_i]; ovs_a += [ov_i]
        t_total_a += batch[0].size(0)

    f1s_a = np.array(f1s_a); ovs_a = np.array(ovs_a)
    return t_total_a, f1s_a, ovs_a, t_a

def main():
    print('[program starts.]')
    args.data_file = 'SQuAD/data_a2.msgpack'
    train, dev, dev_y, train_y, embedding, opt, q_labels, ql_mask = utils.load_data(vars(args), args)
    
    if args.resume:
        print('[loading previous model...]')
        checkpoint = torch.load(os.path.join(model_dir, args.restore_dir, args.resume))
        if args.resume_options:
            opt = checkpoint['config']
        state_dict = checkpoint['state_dict']
        model = DocReaderModel(opt, embedding, state_dict)
    else:
        raise RuntimeError('Include checkpoint of the trained model')   

    if args.cuda:
        model.cuda()

    inputs = [train, dev, dev_y, train_y, embedding, opt, q_labels, ql_mask]
    t_total_a2, f1s_a2, ovs_a2, t_a2 = accuracies_on_ds('SQuAD/data_a2.msgpack', inputs, model, 2)

    args.data_file = 'SQuAD/data_a3.msgpack'
    train, dev, dev_y, train_y, embedding, opt, q_labels, ql_mask = utils.load_data(vars(args), args)
    inputs = [train, dev, dev_y, train_y, embedding, opt, q_labels, ql_mask]
    t_total_a3, f1s_a3, ovs_a3, t_a3 = accuracies_on_ds('SQuAD/data_a3.msgpack', inputs, model, 3)

    args.data_file = 'SQuAD/data_a1.msgpack'
    train, dev, dev_y, train_y, embedding, opt, q_labels, ql_mask = utils.load_data(vars(args), args)
    inputs = [train, dev, dev_y, train_y, embedding, opt, q_labels, ql_mask]
    t_total_a1, f1s_a1, ovs_a1, t_a1 = accuracies_on_ds('SQuAD/data_a1.msgpack', inputs, model, 1)


    def toscore(score, total):
        d = {}
        for p,s in score.items():
            d[p] = round(100.*s/total, 2)
        td = OrderedDict(sorted(d.items(), key=lambda t: t[0]))
        return td

    print("ratio |a|=1: ", toscore(t_a1, t_total_a1), t_total_a1)
    print("ratio |a|=2: ", toscore(t_a2, t_total_a2), t_total_a2)
    print("ratio |a|=3: ", toscore(t_a3, t_total_a3), t_total_a3)

    def toscore2(score):
        return round(100. * score.sum() / len(score), 2)
    axis = 1
    if args.n_actions > 0:
        print("[max F1_a1: {} F1_a2: {} F1_a3: {}]".format(json.dumps(toscore2(np.max(f1s_a1, axis)),toscore2(np.max(f1s_a2, axis)), toscore2(np.max(f1s_a3, axis)))))
        print("[max RE_a1: {} RE_a2: {} RE_a3: {}]".format(json.dumps(toscore2(np.max(ovs_a1, axis)),toscore2(np.max(ovs_a2, axis)), toscore2(np.max(ovs_a3, axis)))))
    else:
        print("[max F1_a1: {} max F1_a2: {} F1_a3: {}]".format(json.dumps(toscore2(f1s_a1), toscore2(f1s_a2), toscore2(f1s_a3))))
        print("[max RE_a1: {} max RE_a2: {} RE_a3: {}]".format(json.dumps(toscore2(ovs_a1),toscore2(ovs_a2), toscore2(ovs_a3))))


if __name__ == '__main__':
    main()
