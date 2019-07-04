import re
import os
import sys
import random
import argparse
from datetime import datetime

import msgpack, time
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

squad_dir = 'SQuAD'
if args.squad == 2:
    squad_dir = 'SQuAD2'
    if 'data2' in args.data_file:
        args.data_file = 'SQuAD2/data2.msgpack'
    else:
        args.data_file = 'SQuAD2/data.msgpack'

# set model dir
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)
model_dir = os.path.abspath(model_dir)
timestamp = time.strftime("%mm%dd_%H%M%S")
print("timestamp {}".format(timestamp))
current_dir = os.path.join(args.model_dir, timestamp)
os.makedirs(current_dir)
torch.set_printoptions(precision=10)
# save model configuration
s = "\nParameters:\n"
for k in sorted(args.__dict__):
    s += "{} = {} \n".format(k, args.__dict__[k])
with open(os.path.join(args.model_dir, timestamp, "about.txt"),"w") as txtf:
    txtf.write(s); print(s)
# set random seed
seed = args.seed if args.seed >= 0 else int(random.random()*1000)
print ('seed:', seed)
random.seed(seed)
torch.manual_seed(seed)
if args.cuda:
    torch.cuda.manual_seed(seed)

log = utils.setup_logger(__name__, os.path.join(current_dir,args.log_file))

def main():
    log.info('[program starts.]')
    train, dev, dev_y, train_y, embedding, opt, q_labels, ql_mask = utils.load_data(vars(args), args)
    log.info('[Data loaded.ql_mask]')

    if args.resume:
        log.info('[loading previous model...]')
        checkpoint = torch.load(os.path.join(model_dir, args.restore_dir, args.resume))
        if args.resume_options:
            opt = checkpoint['config']
        state_dict = checkpoint['state_dict']
        model = DocReaderModel(opt, embedding, state_dict)
    else:
        raise RuntimeError('Include checkpoint of the trained model')   

    if args.cuda:
        model.cuda()

    outputs = ""
    # evaluate restored model
    model.opt['interpret'] = False
    batches = utils.BatchGen(dev, batch_size=100, evaluation=True, gpu=args.cuda)
    predictions = []
    for i, batch in enumerate(batches):
        predictions.extend(model.predict(batch)[0])
    em, f1 = utils.score(predictions, dev_y)
    log.info("[dev EM: {} F1: {}]".format(em, f1))
    outputs += "[dev EM: {} F1: {}]\n".format(em, f1)

    with open(os.path.join(squad_dir,'meta.msgpack'), 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')
    vocab = meta['vocab']
    ids_word = {i:w for i,w in enumerate(vocab)}

    def to_text(inp):
        s = ""
        for ids in inp.numpy():
            s += ids_word[ids] + " "
        return s
    test_int = {i:[] for i in range(args.n_actions)} 
    batches = utils.BatchGen(dev, batch_size=args.batch_size, evaluation=True, gpu=args.cuda, shuffle=True)
    for i, batch in enumerate(batches):
        model.opt['interpret'] = False
        # collect predicted answers for various interpretations
        predictions, acts = model.predict_inter(batch)[:2]
        truth = np.take(dev_y, batches.indices[i], 0)
        for b in range(len(predictions)):
            em_v, f1_v = utils.score([predictions[b]], [truth[b]])
            log.warn("b={0} a={1} EM: {2:.3f} F1: {3:3f}".format(b, acts[b], em_v, f1_v))
        model.opt['interpret'] = True
        i_predictions = []
        for a in range(args.n_actions):
            latent_a = Variable(torch.ones(batch[0].size()[0])*a).long().cuda()
            i_predictions.append(model.predict_inter(batch, latent_a=latent_a)[0])

        for b in range(batch[0].size()[0]):
            f1s = []
            for a in range(args.n_actions):
                em_v, f1_v = utils.score([i_predictions[a][b]], [truth[b]])
                f1s.append(f1_v)
            
            if len(set(f1s))>=1:
                outputs += batch[-2][b] + '\n' + to_text(batch[5][b]) + '\n'
                outputs += "pred_a={} truth={}".format(acts[b], truth[b]) + '\n'
                for a in range(args.n_actions):
                    test_int[a] += [i_predictions[a][b]]
                    em_v, f1_v = utils.score([i_predictions[a][b]], [truth[b]])
                    outputs += i_predictions[a][b] + '\n'+ "b={0} a={1} EM: {2:.3f} F1: {3:3f}".format(b, a, em_v, f1_v) + '\n'
                    log.warn("b={0} a={1} EM: {2:.3f} F1: {3:3f}".format(b, a, em_v, f1_v))
                outputs += '\n'

    with open(os.path.join(current_dir, 'ints.msgpack'), 'wb') as f:
        msgpack.dump(test_int, f)

    with open(os.path.join(current_dir, "interpret.txt"),"w") as txtf:
        txtf.write(outputs);
                

if __name__ == '__main__':
    main()
