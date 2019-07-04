import re
import os
import sys
import random
import argparse
from datetime import datetime

import spacy
import msgpack, time
import numpy as np
import multiprocessing
import unicodedata
import collections

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


docs = ['Maria had a big lunch today. She ate a sandwich. Maria ate a salad with coffee. Finally, she wandered into a store and ate an ice cream.',\
        'Parrot have learned how to reproduce human language. The Bird speaks Japanese now. In fact, the parrot speaks Russian too. And of course, british owner taught this bird how to speak English.',\
        'Manager was late for work and his boss was angry about it. It is because at first manager went to a bank. Then manager went to a friends house. Eventually, a manager went to the cafe.',
        'It is well known that dry air is mainly made up of nitrogen (78.09%) and oxygen (20.95%). However, many of us could not imagine that the rest of dry air is made of argon, carbon dioxide and other trace gases (0.94%).',
        'Africa has varied array of wild animals. The giraffes, the world\'s tallest animal, inhabit Africa. Also African elephants live here. The world\'s fastest land mammal, the cheetah, lives in Africa too.',
        'German language is wide spread in Europe. Obviously, it is mainly spoken in Germany. Moreover, it is one of the used languges in Switzerland and Austria as well.',
        'Town A is located 150 km away from town B. The towns are connected via rail system. A journey between these towns takes around 1 hour by train.',
        'Pulp Fiction is a an American crime film by Quentin Tarantino. In the movie Uma Thurman played Mia. Another main role was given to John Travolta. And lastly, Samuel Jackson also played in the movie and it elevated his career.',
        'Bob keeps his postage marks in a case that is green colored. He have been collecting this marks since his childhood. The case is made of wood. The notable thing about it is that it is carved with waves.',
        'Alice was listening to Beatles yesterday. It was a sunny day, and the song \"Come Together\" fitted perfectly. Indeed, that song was very cheerful and bright.', 
        ]

ques = ['What did Maria eat for lunch ?',\
        'What languages does parrot speak?',\
        'Where did manager go before the work?',
        'What does dry air comprise?',
        'What animals live in Africa?',
        'In which countries German language is spoken?',
        'How far are the two towns from each other?',
        'Who took a part in the movie?',
        'How does the Bob\'s case look like?',
        'What kind of song was Alisce listening?']

ans = [['sandwich', 'salad with coffee', 'ice cream'],\
       ['Japanese', 'Russian', 'English'],\
       ['bank', 'friend house', 'cafe'],
       ['nitrogen', 'oxygen', 'argon', 'carbon dioxide', 'trace gases'],
       ['giraffes', 'elephants', 'cheetah'],
       ['Germany', 'Switzerland', 'Austria'],
       ['150 km', '1 hour by train'],
       ['Thurman', 'Jackson', 'Travolta'],
       ['wood', 'green', 'carved with waves'],
       ['Come Together', 'cheerful', 'bright']]

args.batch_size = len(docs)

def pre_proc(text):
    '''normalize spaces in a string.'''
    text = re.sub('\s+', ' ', text)
    return text

def token2id(docs, vocab, unk_id=None):
    w2id = {w: i for i, w in enumerate(vocab)}
    ids = [[w2id[w] if w in w2id else unk_id for w in doc] for doc in docs]
    return ids

def normalize_text(text):
    return unicodedata.normalize('NFD', text)

def load_data(opt, args, contexts, questions, answers):
    # max q len = 60
    # max c len = 767
    if opt['squad'] == 1:
        squad_dir = 'SQuAD'
    else:
        squad_dir = 'SQuAD2'

    with open(os.path.join(squad_dir, 'meta.msgpack'), 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')

    vocab = meta['vocab']
    vocab_ent = meta['vocab_ent']
    ids_word = {i:w for i,w in enumerate(vocab)}
    embedding = torch.Tensor(meta['embedding'])

    opt['pretrained_words'] = True
    opt['vocab_size'] = embedding.size(0)
    opt['embedding_dim'] = embedding.size(1)
    if not opt['fix_embeddings']:
        embedding[1] = torch.normal(means=torch.zeros(opt['embedding_dim']), std=1.)

    nlp = spacy.load('en') 
    context_text = [pre_proc(c) for c in contexts]
    question_text = [pre_proc(q) for q in questions]
    threads = multiprocessing.cpu_count()
    context_docs = [doc for doc in nlp.pipe(
        iter(context_text), batch_size=64, n_threads=threads)]
    question_docs = [doc for doc in nlp.pipe(
        iter(question_text), batch_size=64, n_threads=threads)]

    question_tokens = [[normalize_text(w.text) for w in doc] for doc in question_docs]
    context_tokens = [[normalize_text(w.text) for w in doc] for doc in context_docs]

    context_token_span = [[(w.idx, w.idx + len(w.text)) for w in doc] for doc in context_docs]
    context_tags = [[w.tag_ for w in doc] for doc in context_docs]
    context_ents = [[w.ent_type_ for w in doc] for doc in context_docs]
    context_features = []
    for question, context in zip(question_docs, context_docs):
        question_word = {w.text for w in question}
        question_lower = {w.text.lower() for w in question}
        question_lemma = {w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() for w in question}
        match_origin = [w.text in question_word for w in context]
        match_lower = [w.text.lower() in question_lower for w in context]
        match_lemma = [(w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower()) in question_lemma for w in context]
        context_features.append(list(zip(match_origin, match_lower, match_lemma)))
    log.info('tokens generated')

    question_ids = token2id(question_tokens, vocab, unk_id=1)
    context_ids = token2id(context_tokens, vocab, unk_id=1)

    context_tf = []
    for doc in context_tokens:
        counter_ = collections.Counter(w.lower() for w in doc)
        total = sum(counter_.values())
        context_tf.append([counter_[w.lower()] / total for w in doc])
    context_features = [[list(w) + [tf] for w, tf in zip(doc, tfs)] for doc, tfs in
                        zip(context_features, context_tf)]

    context_tags = [[w.tag_ for w in doc] for doc in context_docs]    
    vocab_tag = list(nlp.tagger.tag_names)
    context_tag_ids = token2id(context_tags, vocab_tag)
    context_ent_ids = token2id(context_ents, vocab_ent)
    ans_exists = [1]*len(contexts)

    dev = list(zip(
        context_ids,
        context_features,
        context_tag_ids,
        context_ent_ids,
        question_ids,
        ans_exists,
        context_text,
        context_token_span
    ))
    dev_y = answers
    return dev, dev_y, embedding, opt 

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
    dev, dev_y, embedding, opt = load_data(vars(args), args, docs, ques, ans)
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
    
    batches = utils.BatchGen(dev, batch_size=len(docs), evaluation=True, gpu=args.cuda)
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
            latent_a = Variable(torch.ones(args.batch_size)*a).long().cuda()
            i_predictions.append(model.predict_inter(batch, latent_a=latent_a)[0])

        for b in range(args.batch_size):
            f1s = []
            for a in range(args.n_actions):
                em_v, f1_v = utils.score([i_predictions[a][b]], [truth[b]])
                f1s.append(f1_v)
            outputs += batch[-2][b] + '\n' + to_text(batch[5][b]) + '\n'
            outputs += "pred_a={} truth={}".format(acts[b], truth[b]) + '\n'
            for a in range(args.n_actions):
                outputs += i_predictions[a][b] + '\n'
            outputs += '\n'
    print(outputs)        

if __name__ == '__main__':
    main()
