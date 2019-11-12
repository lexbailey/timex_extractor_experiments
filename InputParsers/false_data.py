from StructInflate import Inflator
from os import path
import sys
import random
from itertools import chain

def _rand_fake_data():
    fake_nothing = [{'token':'nothing'}]
    fake_nothing1 = [{'token':'something'}]
    fake_nothing2 = [{'token':'thing'}]
    fake_nothing3 = [{'token':'cat'}]
    fake_nothing4 = [{'token':'test'}]
    _yes_timex = {'token':'timex', 'timexes':[{}]}
    _no_timex = {'token':'timex'}
    timex_yes = [_yes_timex, {'token': 'at'}]
    timex_no = [_no_timex, {'token': 'dog'}]
    tok = lambda: random.choice([timex_yes, timex_no, fake_nothing, fake_nothing1, fake_nothing2, fake_nothing3, fake_nothing4])
    #tok = lambda: random.choice([fake_event, timex_yes, fake_nothing])
    return list(chain(*[tok() for i in range(random.randrange(3,10))]))

def parse(folder, enc='utf-8'):
    return {'a':[
        _rand_fake_data() for i in range(999)  ]}

def sentences(data):
    for k in data:
        for s in data[k]:
            yield s

def reconstruct_sentence(data, fid, sid, tokens=False):
    s = data[fid][sid]
    parts = []
    print(s)
    for i in s:
        if list(i.keys()) == ['token']:
            parts.append(i['token'])
        else:
            if tokens:
                parts.append(i.get('token', '<?>') + str(i))
            else:
                parts.append(i.get('token', '<?>'))
    return ' '.join(parts)


if __name__ == '__main__':
    data = parse('./data/tempeval-training-2/english/data')
    print(data)
    print(reconstruct_sentence(data, next(iter(data.keys())), 2))
