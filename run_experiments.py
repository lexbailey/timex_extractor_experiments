#!/usr/bin/env python
import argparse

from timex_finder_regex import run_experiment as e1
from timex_finder_bayes import run_experiment as e2
from timex_finder_deep_nn import run_experiment as e3

parser = argparse.ArgumentParser()
parser.add_argument('-k', type=int, default=5, help='value of K for K-fold crossvalidation')
parser.add_argument('--only', type=int, help='only run the experiment with this id [0=regex, 1=bayes, 2=deep-nn]')
args = parser.parse_args()

available_experiments = [e1, e2, e3]

if args.only:
    try:
        experiments = [available_experiments[args.only]]
    except IndexError:
        print("Can't find experiment number %d (--only)" % (args.only))
else:
    experiments = available_experiments

k = args.k

results = []
for e in experiments:
    result = e(k)
    assert isinstance(result, list)
    results.extend(result)

for result in results:
    print(result['name'])
    print('\t' + '\n\t'.join('%s: %.2f' % (metric, result[metric]*100) for metric in ['p','r','f']))
    print('\t' + str(result['cross']))
