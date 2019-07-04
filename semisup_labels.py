from sklearn.cluster import KMeans
import argparse
import sent2vec
import msgpack
import pickle
import pandas as pd
from os.path import join

# number of interpretations:         
#      n_labels = [2, 3, 4, 5, 8, 10]
# ratio of labelled interpretations: 
#      ratios = [i*0.01 for i in [30, 35, 35, 45, 45, 50]]
parser = argparse.ArgumentParser(
    description='Preprocessing data files, about 10 minitues to run.'
)
parser.add_argument('--squad', default=1, type=int,help='SQuAD type: 1.0 or 2.0')
args = parser.parse_args()
if args.squad == 1:
    squad_dir = 'SQuAD'
else:
    squad_dir = 'SQuAD2'


n_labels = [2, 3, 4, 5]
ratios = [i*0.01 for i in [30, 35, 35, 45]]


def ids_to_word(s, vocab):
	res = []
	for k in s:
		res.append(vocab.get(k, '<UNK>'))
	return " ".join(res)

# load ids and vocab
with open(join(squad_dir,'meta.msgpack'), 'rb') as f:
    meta = msgpack.load(f, encoding='utf8')
vocab = meta['vocab']
id2word = {k:vocab[k] for k in range(len(vocab))}
word2id = {vocab[k]:k for k in range(len(vocab))}
with open(join(squad_dir,'data.msgpack'), 'rb') as f:
    data = msgpack.load(f, encoding='utf8')
print("Data loaded")

# make sentences from ids
train_orig = pd.read_csv(join(squad_dir,'train.csv'))
q_ids = data['trn_question_ids']
text = data['trn_context_text']
s_idx = train_orig['answer_start_token'].tolist()
e_idx = train_orig['answer_end_token'].tolist()
spans = data['trn_context_spans']
exists = data['trn_ans_exists']

questions_answers = []
for i, (sent_id,t) in enumerate(zip(q_ids, text)):
	if args.squad == 2 and exists[i] == 0:
		questions_answers.append("")
		continue
	sent1 = ids_to_word(sent_id, id2word)
	s_offset, e_offset = spans[i][s_idx[i]][0], spans[i][e_idx[i]][1]
	sent2 = t[s_offset:e_offset]
	questions_answers.append(sent1+' '+sent2)

# convert sentences to embeddings
model = sent2vec.Sent2vecModel()
model.load_model('sent2vec/wiki_bigrams.bin')
embs = model.embed_sentences(questions_answers)
print("Sent2vec loaded")

result = {n_l:[0,0] for n_l in n_labels}
for n_l, ratio in zip(n_labels, ratios):
	kmeans = KMeans(n_clusters=n_l, init='k-means++', max_iter=300, n_init=15)
	kmeans.fit(embs)
	print("labels = %d"%n_l)
	q_l = []
	for l in kmeans.labels_[:int(len(questions_answers)*ratio)+1]:
		q_l.append(l)
	q_l += [0]*(len(questions_answers) - int(len(questions_answers)*ratio)-1)
	mask_l = [1]*(int(len(questions_answers)*ratio)+1) + [0]*(len(questions_answers) - int(len(questions_answers)*ratio)-1)
	assert len(q_l)==len(questions_answers) and len(mask_l)==len(questions_answers), "labels length"
	result[n_l] = [q_l, mask_l]

result.update({"q_text": questions_answers})
with open(join(squad_dir,'q_labels_sm5.pickle'), 'wb') as f:
	pickle.dump(result, f)


