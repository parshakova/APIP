""" get scores for F1 Threshold(rho) experiments for competitive approaches """
# requires: 
#   - json file with predicted answers
#   - json data file


from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys

import numpy as np
from collections import OrderedDict


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
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


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'EM': round(exact_match, 2), 'F1': round(f1, 2)}



def evaluate_dev_flat(dataset, predictions):
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                ans_set = set(ground_truths)
                for a_i in range(len(set(ground_truths))):
                        if qa['answers'][a_i]['text'] in ans_set:
                                 ans_set.remove(qa['answers'][a_i]['text'])
                                 total += 1
                        else:
                                 continue
                        gt_ans = [qa['answers'][a_i]['text']]
                        prediction = predictions[qa['id']]
                        exact_match += metric_max_over_ground_truths(
                            exact_match_score, prediction, gt_ans)
                        f1 += metric_max_over_ground_truths(
                            f1_score, prediction, gt_ans)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'EM': round(exact_match, 2), 'F1': round(f1, 2)}


def toscore(score, total):
    d = {}
    for p,s in score.items():
        d[p] = round(100.*s/total, 2)
    td = OrderedDict(sorted(d.items(), key=lambda t: t[0]))
    return td

def evaluate_dev_a(dataset, predictions, n_ans):
    f1 = exact_match = total = 0
    t_a = {0.1:0, 0.2:0, 0.3:0, 0.4:0, 0.5:0, 0.6:0, 0.7:0, 0.8:0, 0.9:0}
    f1_all = []
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                ans_set = set(ground_truths)
                if len(ans_set) < n_ans or (n_ans == 1 and len(ans_set)>1):
                    continue
                ans_set = list(ans_set)[:n_ans]
                f1s = []; total += 1
                prediction = predictions[qa['id']]
                for a in ans_set:
                    f1s += [metric_max_over_ground_truths(f1_score, prediction, [a])]

                f1_all += [max(f1s)]
                f1s = np.array(f1s)
                for p in t_a.keys():
                    t_a[p] = t_a[p] + int((f1s>p).sum() == n_ans)


    f1s_all = round(100. * sum(f1_all) / len(f1_all), 2)

    t_ans = toscore(t_a, total)

    return t_ans, total, f1s_all


if __name__ == '__main__':
    expected_version = '1.1'
    parser = argparse.ArgumentParser(
        description='Evaluation for SQuAD ' + expected_version)
    parser.add_argument('dataset_file', help='Dataset file')
    parser.add_argument('prediction_file', help='Prediction File')
    args = parser.parse_args()
    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        if (dataset_json['version'] != expected_version):
            print('Evaluation expects v-' + expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)

    print("Orig dev set", json.dumps(evaluate(dataset, predictions)))
    print("Flat dev set", json.dumps(evaluate_dev_flat(dataset, predictions)))

    print("Dev |a|=1", json.dumps(evaluate_dev_a(dataset, predictions, 1)))
    print("Dev |a|=2", json.dumps(evaluate_dev_a(dataset, predictions, 2)))
    print("Dev |a|=3", json.dumps(evaluate_dev_a(dataset, predictions, 3)))