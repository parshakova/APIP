import re
import os
import sys
import random
import argparse
import json
from datetime import datetime

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
if args.squad == 2:
    if 'data2' in args.data_file:
        args.data_file = 'SQuAD2/data2.msgpack'
    else:
        args.data_file = 'SQuAD2/data.msgpack'

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

def main():
    print('[program starts.]')
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

    with open(args.data_file, 'rb') as f:
        data = msgpack.load(f, encoding='utf8')
    dev_ids = data['dev_ids']

    # evaluate restored model
    model.opt['interpret'] = False
    batches = utils.BatchGen(dev, batch_size=args.batch_size, evaluation=True, gpu=args.cuda)
    predictions = []
    for i, batch in enumerate(batches):
        predictions.extend(model.predict(batch)[0])
    em, f1 = utils.score(predictions, dev_y)
    print("[sampled EM: {} F1: {}]".format(em, f1))

    batches = utils.BatchGen(dev, batch_size=args.batch_size, evaluation=True, gpu=args.cuda)
    model.opt['interpret'] = True
    t_em_c, t_f1_c, t_total = [0]*3
    f1s = []
    ems = []
    pred_answers = {}
    # evaluate the model for all interpretations and select the one with highest accuracy
    for i, batch in tqdm(enumerate(batches)):
        i_predictions = []
        truth = np.take(dev_y, batches.indices[i], 0)
        confidence = []; ans_a = []
        for a in range(args.n_actions):
            latent_a = Variable(torch.ones(batch[0].size(0))*a).long().cuda()
            pred = model.predict_inter(batch, latent_a=latent_a)
            i_predictions.append(pred[0])
            computed_a = pred[-1]
            confidence.append(pred[-2])
            ans_a += [pred[0]]

        confidence = np.array(confidence)
        for b in range(batch[0].size(0)):
            em_v, f1_v = [], []
            a = np.argmax(confidence[:,b]) 
            em_c, f1_c = utils.score_test_alli([i_predictions[a][b]], [truth[b]])

            for a in range(args.n_actions):
                em_a, f1_a = utils.score_test_alli([i_predictions[a][b]], [truth[b]])
                em_v += [em_a]
                f1_v += [f1_a]

            pred_answers[dev_ids[i*args.batch_size+b]] = [[a_i[b] for a_i in ans_a], list(map(str, f1_v)), str(computed_a[b])]

            f1s += [f1_v]
            ems += [em_v]
            t_em_c += em_c; t_f1_c += f1_c

        t_total += batch[0].size(0)
        
    with open('predictions_a.json', 'w') as f:
        json.dump(pred_answers, f)

    def toscore(score):
        return 100. * score / t_total

    f1s = np.array(f1s); ems = np.array(ems)

    print("[max EM: {} F1: {}]".format(toscore(np.max(ems, 1).sum()), toscore(np.max(f1s, 1).sum())))
    print("[min EM: {} F1: {}]".format(toscore(np.min(ems, 1).sum()), toscore(np.min(f1s, 1).sum())))
    print("[avg EM: {} F1: {}]".format(toscore(np.average(ems, 1).sum()), toscore(np.average(f1s, 1).sum())))

    print("[con EM: {} F1: {}]".format(toscore(t_em_c), toscore(t_f1_c)))
                

if __name__ == '__main__':
    main()
