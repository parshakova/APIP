import re
import json
import spacy
import argparse
import collections
import msgpack
import logging
import unicodedata
import random
import string
from os.path import join

import numpy as np
import pandas as pd

import multiprocessing
from itertools import compress
from concurrent.futures import ProcessPoolExecutor


parser = argparse.ArgumentParser(
    description='Preprocessing data files, about 10 minitues to run.'
)
parser.add_argument('--wv_file', default='glove/glove.840B.300d.txt',
                    help='path to word vector file.')
parser.add_argument('--wv_dim', type=int, default=300,
                    help='word vector dimension.')
parser.add_argument('--wv_cased', type=bool, default=True,
                    help='treat the words as cased or not.')
parser.add_argument('--sort_all', action='store_true',
                    help='sort the vocabulary by frequencies of all words. '
                         'Otherwise consider question words first.')
parser.add_argument('--sample_size', type=int, default=0,
                    help='size of sample data (for debugging).')
parser.add_argument('--threads', type=int, default=multiprocessing.cpu_count(),
                    help='number of threads for preprocessing.')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size for multiprocess tokenizing and tagging.')

parser.add_argument('--squad', default=1, type=int,help='SQuAD type: 1.0 or 2.0')

args = parser.parse_args()
if args.squad == 1:
    squad_dir = 'SQuAD'
    trn_file = 'SQuAD/train-v1.1.json'
    dev_file = 'SQuAD/dev-v1.1.json'
else:
    squad_dir = 'SQuAD2'
    trn_file = 'SQuAD2/train-v2.0.json'
    dev_file = 'SQuAD2/dev-v2.0.json'

wv_file = args.wv_file
wv_dim = args.wv_dim
val_size = 5000


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG,
                    datefmt='%m/%d/%Y %I:%M:%S')
log = logging.getLogger(__name__)

log.info('start data preparing...')


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



def normalize_text(text):
    return unicodedata.normalize('NFD', text)


def load_wv_vocab(file):
    '''Load tokens from word vector file.

    Only tokens are loaded. Vectors are not loaded at this time for space efficiency.

    Args:
        file (str): path of pretrained word vector file.

    Returns:
        set: a set of tokens (str) contained in the word vector file.
    '''
    vocab = set()
    with open(file) as f:
        for line in f:
            elems = line.split()
            token = normalize_text(''.join(elems[0:-wv_dim]))  # a token may contain space
            vocab.add(token)
    return vocab
wv_vocab = load_wv_vocab(wv_file)
log.info('glove loaded.')


def flatten_json(file, proc_func):
    '''A multi-processing wrapper for loading SQuAD data file.'''
    with open(file) as f:
        data = json.load(f)['data']
    with ProcessPoolExecutor(max_workers=args.threads) as executor:
        rows = executor.map(proc_func, data)
    rows = sum(rows, [])
    return rows


def proc_train(article):
    '''Flatten each article in training data.'''
    rows = []; exists = 1; 
    for paragraph in article['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            id_, question, answers = qa['id'], qa['question'], qa['answers']
            if args.squad == 2:
                exists = 1 - int(qa['is_impossible'])
            if exists == 1:
                ans_all = [answers[a_i]['text'] for a_i in range(len(answers))]; ans_all_set = set(ans_all)
                # when multiple answers per question in the ground truth
                # in SQuAD only one answer per question in training set
                for a_i in range(len(set(ans_all))):
                    if answers[a_i]['text'] in ans_all_set:
                        ans_all_set.remove(answers[a_i]['text'])
                    else:
                        continue
                    answer = answers[a_i]['text']  # in training data there's only one answer
                    answer_start = answers[a_i]['answer_start']
                    answer_end = answer_start + len(answer)
                    rows.append((id_, context, question, answer, int(answer_start), int(answer_end), exists))
            else:
                rows.append((id_, context, question, "", 0, 1, exists))
    
    return rows

def train_add_new_neg(rows):
    _, ts, qs, _, _, _, ae_list = zip(*rows)
    ts_ae = list(compress(ts, ae_list))
    ts_nae = list(compress(ts, 1-np.array(ae_list)))
    # use new documents for negative samples
    new_ts_nae = set(ts_ae) - set(ts_nae)
    t_ids = [i for i, t in enumerate(ts) if t in new_ts_nae]
    tq_dict = {}
    new_rows = []

    # find indices for each text document to use it in negative question selection
    for i, t in enumerate(ts): 
        if t in new_ts_nae:
            if t in tq_dict:
                tq_dict[t] = tq_dict[t] + [i]
            else:
                tq_dict[t] = [i]
    ids_left = set(t_ids)
    print(len(t_ids), len(ids_left), len(new_ts_nae), len(ts_nae), len(ts_ae))

    for t_nae in new_ts_nae:
        iters = 3 + np.random.randint(2)
        q_nae_total = ids_left - set(tq_dict[t_nae])
        try:
            ids_sampled = set(random.sample(q_nae_total, iters))
        except:
            print(len(ids_left), iters, len(q_nae_total),set(tq_dict[t_nae]))
        for i in ids_sampled:
            new_rows.append((len(rows)+len(new_rows), t_nae, qs[i], "", 0, i, 0))
        ids_left = ids_left - ids_sampled

    # verification of new_rows list
    for elem in new_rows:
        assert not(elem[-2] in tq_dict[elem[1]]), print(elem[-2], tq_dict[elem[1]], elem[2], "double trouble") 

    rows = rows + new_rows
    return rows


def proc_dev(article):
    '''Flatten each article in dev data'''
    rows = []; exists = 1
    for paragraph in article['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            if args.squad == 2:
                exists = 1 - int(qa['is_impossible'])
            id_, question, answers = qa['id'], qa['question'], qa['answers']
            answers = [a['text'] for a in answers]
            rows.append((id_, context, question, answers, exists))
    return rows


def proc_dev2(article):
    '''Flatten each article in dev data, account for multiple answers in GT'''
    rows = []; exists = 1
    for paragraph in article['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            if args.squad == 2:
                exists = 1 - int(qa['is_impossible'])
            id_, question, answers = qa['id'], qa['question'], qa['answers']
            if exists == 1:
                ans_all = [answers[a_i]['text'] for a_i in range(len(answers))]; 
                #ans_all = [_normalize_answer(answers[a_i]['text']) for a_i in range(len(answers))]; 
                ans_all_set = set(ans_all)
                # when multiple answers per question in the ground truth
                for a_i in list(ans_all_set):
                    answer = a_i
                    rows.append((id_, context, question, [answer], exists))
            else:
                rows.append((id_, context, question, [""], exists))

    return rows

def proc_dev_a1(article):
    '''use only such samples that contain >=1 different answers'''
    rows = []; exists = 1
    for paragraph in article['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            if args.squad == 2:
                exists = 1 - int(qa['is_impossible'])
            id_, question, answers = qa['id'], qa['question'], qa['answers']
            ans_all = [answers[a_i]['text'] for a_i in range(len(answers))]; 
            ans_all_set = set(ans_all)
            if len(ans_all_set) > 1:
                continue
            answers = [a for a in list(ans_all_set)]
            rows.append((id_, context, question, answers, exists))
    return rows

def proc_dev_a2(article):
    '''use only such samples that contain >=2 different answers'''
    rows = []; exists = 1
    for paragraph in article['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            if args.squad == 2:
                exists = 1 - int(qa['is_impossible'])
            id_, question, answers = qa['id'], qa['question'], qa['answers']
            ans_all = [answers[a_i]['text'] for a_i in range(len(answers))]; 
            ans_all_set = set(ans_all)
            if len(ans_all_set) < 2:
                continue
            answers = [a for a in list(ans_all_set)[:2]]
            rows.append((id_, context, question, answers, exists))
    return rows

def proc_dev_a3(article):
    '''use only such samples that contain >=3 different answers'''
    rows = []; exists = 1
    for paragraph in article['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            if args.squad == 2:
                exists = 1 - int(qa['is_impossible'])
            id_, question, answers = qa['id'], qa['question'], qa['answers']
            ans_all = [answers[a_i]['text'] for a_i in range(len(answers))]; 
            ans_all_set = set(ans_all)
            if len(ans_all_set) < 3:
                continue
            answers = [a for a in list(ans_all_set)[:3]]
            rows.append((id_, context, question, answers, exists))
    return rows

train = flatten_json(trn_file, proc_train)
tr_train = flatten_json(trn_file[:-val_size], proc_train)
tr_validation = flatten_json(trn_file[-val_size:], proc_train)
if args.squad == 2:
    train = train_add_new_neg(train)
    tr_train = train_add_new_neg(tr_train)
    tr_validation = train_add_new_neg(tr_validation)
dev = flatten_json(dev_file, proc_dev)
dev2 = flatten_json(dev_file, proc_dev2)
dev_a1 = flatten_json(dev_file, proc_dev_a1)
dev_a2 = flatten_json(dev_file, proc_dev_a2)
dev_a3 = flatten_json(dev_file, proc_dev_a3)

train = pd.DataFrame(train,
                     columns=['id', 'context', 'question', 'answer',
                              'answer_start', 'answer_end', 'exists'])
tr_train = pd.DataFrame(tr_train,
                     columns=['id', 'context', 'question', 'answer',
                              'answer_start', 'answer_end', 'exists'])
tr_validation = pd.DataFrame(tr_validation,
                     columns=['id', 'context', 'question', 'answer',
                              'answer_start', 'answer_end', 'exists'])
print('initial size of training set', train.size, len(train))
dev = pd.DataFrame(dev,
                   columns=['id', 'context', 'question', 'answers', 'exists'])
dev2 = pd.DataFrame(dev2,
                   columns=['id', 'context', 'question', 'answers', 'exists'])
dev_a2 = pd.DataFrame(dev_a2,
                   columns=['id', 'context', 'question', 'answers', 'exists'])
dev_a3 = pd.DataFrame(dev_a3,
                   columns=['id', 'context', 'question', 'answers', 'exists'])
dev_a1 = pd.DataFrame(dev_a1,
                   columns=['id', 'context', 'question', 'answers', 'exists'])
log.info('json data flattened.')

nlp = spacy.load('en', parser=False, tagger=False, entity=False)

def pre_proc(text):
    '''normalize spaces in a string.'''
    text = re.sub('\s+', ' ', text)
    return text
context_iter = (pre_proc(c) for c in train.context)
context_tokens = [[w.text for w in doc] for doc in nlp.pipe(
    context_iter, batch_size=args.batch_size, n_threads=args.threads)]
log.info('got intial tokens.')


def get_answer_index(context, context_token, answer_start, answer_end, exists):
    '''
    Get exact indices of the answer in the tokens of the passage,
    according to the start and end position of the answer.

    Args:
        context (str): the context passage
        context_token (list): list of tokens (str) in the context passage
        answer_start (int): the start position of the answer in the passage
        answer_end (int): the end position of the answer in the passage
        exists (int): whether answer exists or not

    Returns:
        (int, int): start index and end index of answer
    '''
    p_str = 0
    p_token = 0
    while p_str < len(context):
        if re.match('\s', context[p_str]):
            p_str += 1
            continue
        token = context_token[p_token]
        token_len = len(token) # is answer does not exist or if token from character isnt equal to word
        if context[p_str:p_str + token_len] != token:
             #print('1x', context[answer_start:answer_end])
             return (None, None)
            
        if exists == 0:
            return (0, 1)
        if p_str == answer_start:
            t_start = p_token
        p_str += token_len
        if p_str == answer_end:
            try:
                return (t_start, p_token)
            except UnboundLocalError as e:
                #print('2x', context[answer_start:answer_end])
                return (None, None)
        p_token += 1
    #print('3x', context[answer_start:answer_end])
    return (None, None)
train['answer_start_token'], train['answer_end_token'] = \
    zip(*[get_answer_index(a, b, c, d, e) for a, b, c, d, e in
          zip(train.context, context_tokens,
              train.answer_start, train.answer_end, train.exists)])

tr_train['answer_start_token'], tr_train['answer_end_token'] = \
zip(*[get_answer_index(a, b, c, d, e) for a, b, c, d, e in
      zip(tr_train.context, context_tokens,
          tr_train.answer_start, tr_train.answer_end, tr_train.exists)])

tr_validation['answer_start_token'], tr_validation['answer_end_token'] = \
zip(*[get_answer_index(a, b, c, d, e) for a, b, c, d, e in
      zip(tr_validation.context, context_tokens,
          tr_validation.answer_start, tr_validation.answer_end, tr_validation.exists)])

initial_len = len(train)
train.dropna(inplace=True)
tr_train.dropna(inplace=True)
tr_validation.dropna(inplace=True)
print('train size again', train.size, len(train))
log.info('drop {} inconsistent samples.'.format(initial_len - len(train)))
log.info('answer pointer generated.')

questions =list(train.question) + list(dev.question) + list(dev2.question) + list(dev_a2.question) + list(dev_a3.question) + list(dev_a1.question)
contexts = list(train.context) + list(dev.context) + list(dev2.context) + list(dev_a2.context) + list(dev_a3.context) + list(dev_a1.context)

nlp = spacy.load('en') 
context_text = [pre_proc(c) for c in contexts]
question_text = [pre_proc(q) for q in questions]
question_docs = [doc for doc in nlp.pipe(
    iter(question_text), batch_size=args.batch_size, n_threads=args.threads)]
context_docs = [doc for doc in nlp.pipe(
    iter(context_text), batch_size=args.batch_size, n_threads=args.threads)]
if args.wv_cased:
    question_tokens = [[normalize_text(w.text) for w in doc] for doc in question_docs]
    context_tokens = [[normalize_text(w.text) for w in doc] for doc in context_docs]
else:
    question_tokens = [[normalize_text(w.text).lower() for w in doc] for doc in question_docs]
    context_tokens = [[normalize_text(w.text).lower() for w in doc] for doc in context_docs]
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


def build_vocab(questions, contexts):
    '''
    Build vocabulary sorted by global word frequency, or consider frequencies in questions first,
    which is controlled by `args.sort_all`.
    '''
    if args.sort_all:
        counter = collections.Counter(w for doc in questions + contexts for w in doc)
        vocab = sorted([t for t in counter if t in wv_vocab], key=counter.get, reverse=True)
    else:
        counter_q = collections.Counter(w for doc in questions for w in doc)
        counter_c = collections.Counter(w for doc in contexts for w in doc)
        counter = counter_c + counter_q
        vocab = sorted([t for t in counter_q if t in wv_vocab], key=counter_q.get, reverse=True)
        vocab += sorted([t for t in counter_c.keys() - counter_q.keys() if t in wv_vocab],
                        key=counter.get, reverse=True)
    total = sum(counter.values())
    matched = sum(counter[t] for t in vocab)
    log.info('vocab coverage {1}/{0} | OOV occurrence {2}/{3} ({4:.4f}%)'.format(
        len(counter), len(vocab), (total - matched), total, (total - matched) / total * 100))
    vocab.insert(0, "<PAD>")
    vocab.insert(1, "<UNK>")
    with open(join(squad_dir,'vocab.msgpack'), 'rb') as f:
        voc1 = msgpack.load(f, encoding='utf8')
    if not set(vocab).issubset(set(voc1)):
        print("The checkpoints cannot be used, since a vocabulary has different set of words")
        print("New dataset %d vs %d (diff in %d), adopting general vocabulary for checkpoints"%(len(vocab), len(voc1), len(vocab) - len(set(vocab)&set(voc1))))
        vocab = voc1
    else:
        # to preserve the order of indices due to sorted Counter
        vocab = voc1
    return vocab, voc1, counter


def token2id(docs, vocab, unk_id=None):
    w2id = {w: i for i, w in enumerate(vocab)}
    ids = [[w2id[w] if w in w2id else unk_id for w in doc] for doc in docs]
    return ids
vocab, voc1, counter = build_vocab(question_tokens, context_tokens)
# tokens
question_ids = token2id(question_tokens, vocab, unk_id=1)
context_ids = token2id(context_tokens, vocab, unk_id=1)
# term frequency in document
context_tf = []
for doc in context_tokens:
    counter_ = collections.Counter(w.lower() for w in doc)
    total = sum(counter_.values())
    context_tf.append([counter_[w.lower()] / total for w in doc])
context_features = [[list(w) + [tf] for w, tf in zip(doc, tfs)] for doc, tfs in
                    zip(context_features, context_tf)]
# tags
vocab_tag = list(nlp.tagger.tag_names)
context_tag_ids = token2id(context_tags, vocab_tag)
# entities, build dict on the fly
counter_ent = collections.Counter(w for doc in context_ents for w in doc)
vocab_ent = sorted(counter_ent, key=counter_ent.get, reverse=True)
log.info('Found {} POS tags.'.format(len(vocab_tag)))
log.info('Found {} entity tags: {}'.format(len(vocab_ent), vocab_ent))
context_ent_ids = token2id(context_ents, vocab_ent)
log.info('vocab built.')


def build_embedding(embed_file, targ_vocab, dim_vec):
    vocab_size = len(targ_vocab)
    emb = np.zeros((vocab_size, dim_vec))
    w2id = {w: i for i, w in enumerate(targ_vocab)}
    nonzero = 0
    with open(embed_file) as f:
        for line in f:
            elems = line.split()
            token = normalize_text(''.join(elems[0:-wv_dim]))
            if token in w2id:
                emb[w2id[token]] = [float(v) for v in elems[-wv_dim:]]
                nonzero += 1
    return emb, nonzero
embedding, nonzero = build_embedding(wv_file, vocab, wv_dim)
log.info('got embedding matrix.')
print("vocab size = %d, embed_voc size = %d" % (len(vocab), nonzero))

 
tr_train.to_csv(join(squad_dir,'train.csv'), index=False)
tr_validation.to_csv(join(squad_dir,'valid.csv'), index=False)
dev.to_csv(join(squad_dir,'dev.csv'), index=False)

meta = {
    'vocab': vocab,
    'voc': voc1,
    'embedding': embedding.tolist(),
    'vocab_ent':vocab_ent
}
with open(join(squad_dir,'meta.msgpack'), 'wb') as f:
    msgpack.dump(meta, f)

# size of validation set

result = {
    'trn_question_ids': question_ids[:len(train)-val_size],
    'val_question_ids': question_ids[len(train)-val_size:len(train)],
    'dev_question_ids': question_ids[len(train):len(train)+len(dev)],

    'trn_context_ids': context_ids[:len(train)-val_size],
    'val_context_ids': context_ids[len(train)-val_size:len(train)],
    'dev_context_ids': context_ids[len(train):len(train)+len(dev)],

    'trn_context_features': context_features[:len(train)-val_size],
    'val_context_features': context_features[len(train)-val_size:len(train)],
    'dev_context_features': context_features[len(train):len(train)+len(dev)],

    'trn_context_tags': context_tag_ids[:len(train)-val_size],
    'val_context_tags': context_tag_ids[len(train)-val_size:len(train)],
    'dev_context_tags': context_tag_ids[len(train):len(train)+len(dev)],

    'trn_context_ents': context_ent_ids[:len(train)-val_size],
    'val_context_ents': context_ent_ids[len(train)-val_size:len(train)],
    'dev_context_ents': context_ent_ids[len(train):len(train)+len(dev)],

    'trn_context_text': context_text[:len(train)-val_size],
    'val_context_text': context_text[len(train)-val_size:len(train)],
    'dev_context_text': context_text[len(train):len(train)+len(dev)],

    'trn_context_spans': context_token_span[:len(train)-val_size],
    'val_context_spans': context_token_span[len(train)-val_size:len(train)],
    'dev_context_spans': context_token_span[len(train):len(train)+len(dev)],

    'trn_ans_exists': list(tr_train.exists),
    'val_ans_exists': list(tr_validation.exists),
    'dev_ans_exists': list(dev.exists),

    'dev_ids': list(dev.id)
}

print("Train ds %d dev ds %d dev2 ds %d dev_a2 ds %d dev_a3 ds %d"%(len(train), len(dev), len(dev2), len(dev_a2), len(dev_a3)))

with open(join(squad_dir,'data.msgpack'), 'wb') as f:
    msgpack.dump(result, f)

result = {
    'trn_question_ids': question_ids[:len(train)],
    'dev_question_ids': question_ids[len(train)+len(dev):len(train)+len(dev)+len(dev2)],
    'trn_context_ids': context_ids[:len(train)],
    'dev_context_ids': context_ids[len(train)+len(dev):len(train)+len(dev)+len(dev2)],
    'trn_context_features': context_features[:len(train)],
    'dev_context_features': context_features[len(train)+len(dev):len(train)+len(dev)+len(dev2)],
    'trn_context_tags': context_tag_ids[:len(train)],
    'dev_context_tags': context_tag_ids[len(train)+len(dev):len(train)+len(dev)+len(dev2)],
    'trn_context_ents': context_ent_ids[:len(train)],
    'dev_context_ents': context_ent_ids[len(train)+len(dev):len(train)+len(dev)+len(dev2)],
    'trn_context_text': context_text[:len(train)],
    'dev_context_text': context_text[len(train)+len(dev):len(train)+len(dev)+len(dev2)],
    'trn_context_spans': context_token_span[:len(train)],
    'dev_context_spans': context_token_span[len(train)+len(dev):len(train)+len(dev)+len(dev2)],
    'trn_ans_exists': list(train.exists),
    'dev_ans_exists': list(dev2.exists),
    'dev_answers': list(dev2.answers),
    'dev_ids': list(dev2.id)
}

with open(join(squad_dir,'data2.msgpack'), 'wb') as f:
    msgpack.dump(result, f)

result = {
    'trn_question_ids': question_ids[:len(train)],
    'dev_question_ids': question_ids[len(train)+len(dev)+len(dev2):len(train)+len(dev)+len(dev2)+len(dev_a2)],
    'trn_context_ids': context_ids[:len(train)],
    'dev_context_ids': context_ids[len(train)+len(dev)+len(dev2):len(train)+len(dev)+len(dev2)+len(dev_a2)],
    'trn_context_features': context_features[:len(train)],
    'dev_context_features': context_features[len(train)+len(dev)+len(dev2):len(train)+len(dev)+len(dev2)+len(dev_a2)],
    'trn_context_tags': context_tag_ids[:len(train)],
    'dev_context_tags': context_tag_ids[len(train)+len(dev)+len(dev2):len(train)+len(dev)+len(dev2)+len(dev_a2)],
    'trn_context_ents': context_ent_ids[:len(train)],
    'dev_context_ents': context_ent_ids[len(train)+len(dev)+len(dev2):len(train)+len(dev)+len(dev2)+len(dev_a2)],
    'trn_context_text': context_text[:len(train)],
    'dev_context_text': context_text[len(train)+len(dev)+len(dev2):len(train)+len(dev)+len(dev2)+len(dev_a2)],
    'trn_context_spans': context_token_span[:len(train)],
    'dev_context_spans': context_token_span[len(train)+len(dev)+len(dev2):len(train)+len(dev)+len(dev2)+len(dev_a2)],
    'trn_ans_exists': list(train.exists),
    'dev_ans_exists': list(dev_a2.exists),
    'dev_answers': list(dev_a2.answers),
    }

with open(join(squad_dir,'data_a2.msgpack'), 'wb') as f:
    msgpack.dump(result, f)

result = {
    'trn_question_ids': question_ids[:len(train)],
    'dev_question_ids': question_ids[len(train)+len(dev)+len(dev2)+len(dev_a2):len(train)+len(dev)+len(dev2)+len(dev_a2)+len(dev_a3)],
    'trn_context_ids': context_ids[:len(train)],
    'dev_context_ids': context_ids[len(train)+len(dev)+len(dev2)+len(dev_a2):len(train)+len(dev)+len(dev2)+len(dev_a2)+len(dev_a3)],
    'trn_context_features': context_features[:len(train)],
    'dev_context_features': context_features[len(train)+len(dev)+len(dev2)+len(dev_a2):len(train)+len(dev)+len(dev2)+len(dev_a2)+len(dev_a3)],
    'trn_context_tags': context_tag_ids[:len(train)],
    'dev_context_tags': context_tag_ids[len(train)+len(dev)+len(dev2)+len(dev_a2):len(train)+len(dev)+len(dev2)+len(dev_a2)+len(dev_a3)],
    'trn_context_ents': context_ent_ids[:len(train)],
    'dev_context_ents': context_ent_ids[len(train)+len(dev)+len(dev2)+len(dev_a2):len(train)+len(dev)+len(dev2)+len(dev_a2)+len(dev_a3)],
    'trn_context_text': context_text[:len(train)],
    'dev_context_text': context_text[len(train)+len(dev)+len(dev2)+len(dev_a2):len(train)+len(dev)+len(dev2)+len(dev_a2)+len(dev_a3)],
    'trn_context_spans': context_token_span[:len(train)],
    'dev_context_spans': context_token_span[len(train)+len(dev)+len(dev2)+len(dev_a2):len(train)+len(dev)+len(dev2)+len(dev_a2)+len(dev_a3)],
    'trn_ans_exists': list(train.exists),
    'dev_ans_exists': list(dev_a3.exists),
    'dev_answers': list(dev_a3.answers)
}

with open(join(squad_dir,'data_a3.msgpack'), 'wb') as f:
    msgpack.dump(result, f)

result = {
    'trn_question_ids': question_ids[:len(train)],
    'dev_question_ids': question_ids[len(train)+len(dev)+len(dev2)+len(dev_a2)+len(dev_a3):],
    'trn_context_ids': context_ids[:len(train)],
    'dev_context_ids': context_ids[len(train)+len(dev)+len(dev2)+len(dev_a2)+len(dev_a3):],
    'trn_context_features': context_features[:len(train)],
    'dev_context_features': context_features[len(train)+len(dev)+len(dev2)+len(dev_a2)+len(dev_a3):],
    'trn_context_tags': context_tag_ids[:len(train)],
    'dev_context_tags': context_tag_ids[len(train)+len(dev)+len(dev2)+len(dev_a2)+len(dev_a3):],
    'trn_context_ents': context_ent_ids[:len(train)],
    'dev_context_ents': context_ent_ids[len(train)+len(dev)+len(dev2)+len(dev_a2)+len(dev_a3):],
    'trn_context_text': context_text[:len(train)],
    'dev_context_text': context_text[len(train)+len(dev)+len(dev2)+len(dev_a2)+len(dev_a3):],
    'trn_context_spans': context_token_span[:len(train)],
    'dev_context_spans': context_token_span[len(train)+len(dev)+len(dev2)+len(dev_a2)+len(dev_a3):],
    'trn_ans_exists': list(train.exists),
    'dev_ans_exists': list(dev_a1.exists),
    'dev_answers': list(dev_a1.answers)
}

with open(join(squad_dir,'data_a1.msgpack'), 'wb') as f:
    msgpack.dump(result, f)

if args.sample_size:
    sample_size = args.sample_size
    sample = {
        'trn_question_ids': result['trn_question_ids'][:sample_size],
        'dev_question_ids': result['dev_question_ids'][:sample_size],
        'trn_context_ids': result['trn_context_ids'][:sample_size],
        'dev_context_ids': result['dev_context_ids'][:sample_size],
        'trn_context_features': result['trn_context_features'][:sample_size],
        'dev_context_features': result['dev_context_features'][:sample_size],
        'trn_context_tags': result['trn_context_tags'][:sample_size],
        'dev_context_tags': result['dev_context_tags'][:sample_size],
        'trn_context_ents': result['trn_context_ents'][:sample_size],
        'dev_context_ents': result['dev_context_ents'][:sample_size],
        'trn_context_text': result['trn_context_text'][:sample_size],
        'dev_context_text': result['dev_context_text'][:sample_size],
        'trn_context_spans': result['trn_context_spans'][:sample_size],
        'dev_context_spans': result['dev_context_spans'][:sample_size]
    }
    with open(join(squad_dir,'sample.msgpack'), 'wb') as f:
        msgpack.dump(sample, f)
log.info('saved to disk.')
