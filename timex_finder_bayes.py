#!/usr/bin/env python

from InputParsers import tempeval2
#from InputParsers import false_data as tempeval2
import re
from itertools import product, chain, tee
import json
from functools import reduce
from operator import mul
from kfold import ikfold, average_metrics

from bayes_extractor import find_timexes, ProbabilityTable

trial_data_path = './data/tempeval2-trial/data/english'
train_data_path = './data/tempeval-training-2/english/data'

data_train = tempeval2.parse(train_data_path)
data_trial = tempeval2.parse(trial_data_path)

re_time = re.compile(r'(?:\d?\d:\d\d(?:am|pm)?|\d?\d(?:am|pm))$')
re_abs_year = re.compile(r'^(?:(?:\d\d)?\d0\'?s|(?:\d\d\d\d(?:[Aa][Dd]|[Bb][Cc])?|\d?\d?\d?\d(?:[Aa][Dd]|[Bb][Cc])))$')
#re_number = re.compile(r'(?:\d+|a|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|(?:twenty|thirty)(?:one|two|three|four|five|six|seven)?)$')
## Is ordinal the right word here?
#re_ordinal = re.compile(r'(?:this|last|next|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentyth|twentyfirst|twentysecond|tewntythird|twentyfourth|twentyfifth|twentysixth|twentyseventh|twentyeighth|twentyninth|thirtieth|thirtyfirst)$')

cur_regexes = []

hybrid_regexes = [
    re_time
    ,re_abs_year
    #,re_number
    #,re_ordinal
]

def red(s):
    return '\u001b[31m%s\u001b[0m'%s

def green(s):
    return '\u001b[32m%s\u001b[0m'%s
   
def norm_key(k):
    for r in cur_regexes:
        if r.match(k):
            return r
    return k

tables = None

def learn(data):
    data2 = data
    for off in tables:
        data2, data = tee(data2)
        totals = {}
        timexes = {}
        for s in data:
            slen = len(s)
            for i, t in enumerate(s):
                word = norm_key(t['token'].lower().strip())
                pos = i-off
                if pos >= 0 and pos < slen:
                    if word not in totals:
                        totals[word] = 0
                        timexes[word] = 0
                    totals[word] += 1
                    if 'timexes' in s[pos]:
                        timexes[word] += 1
        for word in totals:
            if totals[word] >= 5: # To ensure large enough sample
                p = timexes[word] / totals[word]
                if p > 0.001:
                    tables[off].set(word, p)
    #print(tables)

def div_(a, b):
    try:
        return a/b
    except ZeroDivisionError:
        return float('inf')

def get_weights(data):
    n = 0
    n_ev = 0
    for s in data:
        for t in s:
            if 'timexes' not in t:
                n += 1
            else:
                n_ev += 1
    return div_(n,n_ev)

def evaluate(data, threshold, quiet=False):
    correct = 0
    total = 0
    data, data2 = tee(data)
    w = get_weights(data2)
    totals = {'t':{'n':0,'p':0},'f':{'n':0,'p':0}}
    n_sent = 0
    n_corr = 0
    for s in data:
        n_sent += 1
        text = ' '.join(t['token'] for t in s)
        prediction = find_timexes(s, threshold, tables)
        actual = [1 if 'timexes' in t else 0 for t in s]
        assert len(prediction) == len(actual), "%d != %d" % (len(prediction), len(actual))
        for a, p in zip(actual, prediction):
            truth = 't' if a == p else 'f'
            polarity = 'p' if p == 1 else 'n'
            value = 1 if a == 0 else w
            total += value
            correct += value if a == p else 0
            totals[truth][polarity] += 1
        if all(a == p for a, p in zip(actual, prediction)):
            n_corr += 1
    sacc = div_(n_corr,n_sent)
    print("SAcc %.2f (%d/%d)" % (sacc * 100, n_corr, n_sent))
    score = div_(correct,total)
    print("WAcc %.2f" % (score * 100))
    p = div_(totals['t']['p'], (totals['t']['p'] + totals['f']['p']))
    r = div_(totals['t']['p'], (totals['t']['p'] + totals['f']['n']))
    print("Prec %.2f" % (p * 100))
    print("Rec  %.2f" % (r * 100))
    f = 2*p*r/(p+r)
    print("F    %.2f" % (f * 100))
    return {
        'p': p
        ,'r': r
        ,'f': f
    }

def run_one_experiment(k, regexes, threshold):
    global tables
    global cur_regexes, hybrid_regexes
    results = []
    for train, test in ikfold(tempeval2.sentences(data_train), k):
        tables = {
            i: ProbabilityTable() for i in range(-2, 3)
        }
        cur_regexes = regexes
        learn(train)
        results.append(evaluate(test, threshold))
    return results

def run_experiment(k):

    # Determine threshold
    global tables, current_regexes
    current_regexes = []
    r = range(-2, 3)
    tables = {
        i: ProbabilityTable() for i in r
    }
    learn(tempeval2.sentences(data_train))
    threshold = 0.05
    log = {}
    max_t = 0
    max_f = 0
    while threshold <= 0.951:
        threshold += 0.01
        f = evaluate(tempeval2.sentences(data_trial), threshold)['f']
        log[threshold] = f
        if f > max_f:
            max_f = f
            max_t = threshold
    print(log)
    threshold = max_t
    print("Use threshold of %.3f" % (threshold))

    cross_data_raw = run_one_experiment(k, [], threshold)
    cross_data_hybrid = run_one_experiment(k, hybrid_regexes, threshold)

    return [{
        'name': 'Naive Bayes (standalone), window size = 2+1+2'
        ,'cross':cross_data_raw
        ,**average_metrics(cross_data_raw)
    }
    ,{
        'name': 'Naive Bayes plus RegEx hybrid, window size = 2+1+2'
        ,'cross':cross_data_hybrid
        ,**average_metrics(cross_data_hybrid)
    }]

def export_table():
    global tables
    tables = {
        i: ProbabilityTable() for i in range(-2,3)
    }
    learn(tempeval2.sentences(data_train))
    return tables

if __name__ == '__main__':
    for r in [range(1), range(-1, 2), range(-2, 3), range(-3,4)]:
        tables = {
            i: ProbabilityTable() for i in r
        }
        print("Window range: ", r)
        learn(tempeval2.sentences(data_train))
        baseline_f = evaluate(tempeval2.sentences(data_train), 0.13)
        evaluate(tempeval2.sentences(data_trial), 0.13)
