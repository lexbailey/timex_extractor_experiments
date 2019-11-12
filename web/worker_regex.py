from browser import bind, self

from regex_parser import find_timexes as re_find_timexes
from regex_data import timex_res, anti_timex_res

def run_regex(toks):
    s = [{'token':t} for t in toks]
    result = re_find_timexes(s, timex_res, anti_timex_res, quiet=True)
    return result

@bind(self, "message")
def message(evt):
    c, *data = evt.data
    if c == 'run':
        toks = data[0]
        result = run_regex(toks)
        self.send(['regex', toks, result])

self.send(["loaded_regex"])
