#!/usr/bin/env python
from random import shuffle

def ikfold(dataset, k):
    items = list(dataset)
    shuffle(items)
    n = len(items)
    items_per_group = n // k
    remaining = n - (items_per_group * k)
    last_group = items_per_group + remaining
    groups = ([items_per_group] * (k-1)) + [last_group]
    i = 0
    for g in groups:
        yield items[0:i] + items[i+g:], items[i:i+g]
        i += g

def average_metrics(cross):
    names = list(cross[0].keys())
    n = len(cross)
    totals = {name: 0 for name in names}
    average = {}
    for m in names:
        for c in cross:
            totals[m] += c[m]
        average[m] = totals[m] / n
    return average


if __name__ == '__main__':
    from InputParsers import tempeval2
    import re
    from itertools import product, chain
    import json
    from functools import reduce
    from operator import mul

    data_path = './data/tempeval2-trial/data/english'
    data = tempeval2.sentences(tempeval2.parse(data_path))

    for train, validate in ikfold(data, 5):
        print(len(train), len(validate))

    for train, validate in ikfold([1,2,3,4,5],5):
        print(train, validate)

    print(average_metrics(
        [
            {'a':1, 'b':50}
            ,{'a':0, 'b':60}
        ]
    ))
