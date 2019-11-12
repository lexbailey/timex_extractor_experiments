import json
from itertools import chain

class ProbabilityTable:
    def __init__(self):
        self.probs = {}
        self.rprobs = {}

    def get(self, k, default=None):
        if isinstance(k, str):
            if k in self.probs:
                return self.probs.get(k)
            return self.probs.get(k.lower(), default)
        for r, prob in self.rprobs.items():
            if r.match(k):
                return prob
        return default

    def set(self, k, value):
        if isinstance(k, str):
            self.probs[k.lower()] = value
        else:
            self.rprobs[k] = value

    def __str__(self):
        return '\n'.join([(str(v) + ' < ' + str(k)) for k, v in sorted(chain(self.probs.items(), self.rprobs.items()), key=lambda x: x[1])])

    def __repr__(self):
        return 'ProbabilityTable() # data...\n' + str(self)

    def dump(self):
        assert self.rprobs == {}
        return json.dumps(self.probs)

    def from_json(j):
        t = ProbabilityTable()
        t.probs = json.loads(j)

def find_timexes(s, threshold, tables):
    result = []
    slen = len(s)
    for i, t in enumerate(s):
        probs = []
        for off in tables:
            pos = i+off
            if pos >= 0 and pos < slen:
                word = s[pos]['token']
                prob = tables[off].get(word, 0.0)
                probs.append(prob)
        prob = sum(probs) / len(tables)
        if prob > threshold:
            result.append(1)
        else:
            result.append(0)
    return result

