import re
import os
import sys
import random
import argparse
from datetime import datetime

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import msgpack, time

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
# setup logger
log = utils.setup_logger(__name__, os.path.join(current_dir,args.log_file))

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


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
    
    with open(os.path.join(squad_dir,'meta.msgpack'), 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')
    vocab = meta['vocab']
    ids_word = {i:w for i,w in enumerate(vocab)}

    def to_text(inp):
        s = ""
        for ids in inp.numpy():
            s += ids_word[ids] + " "
        return s

    # evaluate restored model 
    batches = utils.BatchGen(dev, batch_size=100, evaluation=True, gpu=args.cuda)
    predictions = []
    for i, batch in enumerate(batches):
        predictions.extend(model.predict(batch)[0])
    em, f1 = utils.score(predictions, dev_y)
    log.info("[dev EM: {} F1: {}]".format(em, f1))
    

    batches = utils.BatchGen(dev, batch_size=args.batch_size, evaluation=True, gpu=args.cuda)
    model.opt['interpret'] = True
    #itrs = [30, 58]
    itrs = [0,30]
    outputs = ""
    # collect document encodings for induced interpretations (embeds) and interpretations chosen by the model (computed_a)
    X = [[] for _ in range(itrs[1]-itrs[0]+1)]
    for i, batch in enumerate(batches):
        if i < itrs[0]: continue
        truth = np.take(dev_y, batches.indices[i], 0)
        i_predictions = []
        for a in range(args.n_actions):
            latent_a = Variable(torch.ones(args.batch_size)*a).long().cuda()
            i_predictions.append(model.predict_inter(batch, latent_a=latent_a)[0])

        for b in range(len(batch[0])):
            outputs += batch[-2][b] + '\n' + to_text(batch[5][b]) + '\n'
            outputs += "idx = {} truth={}".format((i-itrs[0])*args.batch_size+b, truth[b]) + '\n'
            for a in range(args.n_actions):
                em_v, f1_v = utils.score([i_predictions[a][b]], [truth[b]])
                outputs += i_predictions[a][b] + '\n'+ "b={0} a={1} ".format(i-itrs[0], a, em_v, f1_v) + '\n'
            outputs += '\n'

        for a in range(args.n_actions):
            latent_a = Variable(torch.ones(args.batch_size)*a).long().cuda()
            embeds, actions, questions, computed_a = model.get_embeddings(batch, latent_a=[1,latent_a])
            X[i-itrs[0]].append([embeds, actions, questions, computed_a])
        if i >= itrs[1]:
            break
    
    print(outputs)

    # rearrange encodings 
    x_emb, x_l, x_q, computed_a = [], [], [], []
    for it in range(itrs[1]-itrs[0]+1):
        for b in range(args.batch_size):
            for a in range(args.n_actions): 
                x_emb.append(X[it][a][0][b])
                x_l.append(X[it][a][1][b])
            x_q.append(X[it][a][2][b])
            computed_a.append(X[it][a][3][b])
    x_emb = np.array(x_emb)
    x_l = np.array(x_l)
    x_q = np.array(x_q)
    computed_a = np.array(computed_a).astype(int)

    # 256D -> 2D
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    tsne_d = tsne_model.fit_transform(x_emb)

    # find document encodings for selected interpretations
    a = np.reshape(computed_a, ((itrs[1]-itrs[0]+1)*args.batch_size))
    a_oh = np.expand_dims(np.eye(args.n_actions)[a], -1)
    tsne_d_r = np.reshape(tsne_d, ((itrs[1]-itrs[0]+1)*args.batch_size, args.n_actions, -1))
    sel_tsne_d = np.sum(tsne_d_r*a_oh, 1)

    # setup the plot
    N = args.n_actions

    c = x_l.astype(int)
    x = tsne_d[:, 0]; y = tsne_d[:, 1]
    plt.scatter(x, y, c=c, s=40, cmap=discrete_cmap(N, 'jet'), alpha=0.5)
    names = [str(i//(args.n_actions)) for i in range(tsne_d.shape[0])]
    for i, txt in enumerate(names):
        plt.annotate(txt, (x[i],y[i]), size= 'x-small')

    c = computed_a.astype(int)
    x = sel_tsne_d[:, 0]; y=sel_tsne_d[:, 1]
    plt.scatter(x, y, c=c, s=70,marker='x', cmap=discrete_cmap(N, 'jet'))
    names = [str(i) for i in range(sel_tsne_d.shape[0])]
    for i, txt in enumerate(names):
        plt.annotate(txt, (x[i],y[i]), size= 'x-small')

    plt.colorbar(ticks=range(N))
    plt.clim(-0.5, N - 0.5)
    plt.title("tSNE")
    plt.show()


if __name__ == '__main__':
    main()
