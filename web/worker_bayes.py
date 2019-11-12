from browser import bind, self

from bayes_extractor import find_timexes as bayes_find_timexes
from bayes_data import get_tables as bayes_tables

def run_bayes(toks):
    s = [{'token':t} for t in toks]
    result = bayes_find_timexes(s, 0.13, bayes_tables())
    return result

@bind(self, "message")
def message(evt):
    c, *data = evt.data
    if c == 'run':
        toks = data[0]
        result = run_bayes(toks)
        self.send(['bayes', toks, result])

self.send(["loaded_bayes"])
