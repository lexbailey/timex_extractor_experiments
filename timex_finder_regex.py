#!/usr/bin/env python

from InputParsers import tempeval2
#from InputParsers import false_data as tempeval2
import re
from itertools import product
import json
from regex_parser import find_timexes

trial_data_path = './data/tempeval2-trial/data/english'
train_data_path = './data/tempeval-training-2/english/data'

data_train = tempeval2.parse(train_data_path)
data_trial = tempeval2.parse(trial_data_path)


re_time = re.compile(r'(?:\d?\d:\d\d(?:am|pm)?|\d?\d(?:am|pm))$')
re_rel_day = re.compile(r'(?:today|tomorrow|yesterday)$')
re_abs_day = re.compile(r'(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)$')
#re_abs_month = re.compile(r'(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|sept|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?$')
re_abs_month = re.compile(r'(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|sept|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?$')
re_abs_year = re.compile(r'^(?:(?:\d\d)?\d0\'?s|(?:\d\d\d\d(?:[Aa][Dd]|[Bb][Cc])?|\d?\d?\d?\d(?:[Aa][Dd]|[Bb][Cc])))$')

re_number = re.compile(r'(?:\d+|a|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|(?:twenty|thirty)(?:one|two|three|four|five|six|seven)?)$')
# Is ordinal the right word here?
re_ordinal = re.compile(r'(?:this|last|next|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentyth|twentyfirst|twentysecond|tewntythird|twentyfourth|twentyfifth|twentysixth|twentyseventh|twentyeighth|twentyninth|thirtieth|thirtyfirst)$')

re_period_word = re.compile(r'(?:period|year|quarter|day|month|week|hour|minute|second|decade|milenium|fotnight)s?$')

re_freq = re.compile(r'(?:periodically|anually|daily|yearly|weekly|quarterly|monthly|hourly|fortnightly)')
re_time_word = re.compile(r'(?:noon|dawn|dusk|midday)')

re_others = re.compile(r'(?:now|current|soon|recent)$')

abs_times = [
    [re_abs_month, re_number]
    ,[re_abs_day]
    ,[re_abs_month]
    ,[re_abs_year]
    ,[re_time]
    ,[re_time_word]
]

periods = [
    [re_number, re_period_word]
    ,[re_ordinal, re_period_word]
    ,[re_period_word]
    ,[re_rel_day]
]

others = [
    [re_others]
    ,[re_freq]
]

postfix_1 = re.compile(r'(?:ago|earlier)$')
postfixes = [
    postfix_1
    ,[re.compile(r'ago$'), re.compile(r'period$')]
]
prefix = re.compile(r'(?:a|the|mid|latest|current|early|late|earlier|after|within|for|about|recent|last)$')

from itertools import chain

def join_re(*args):
    lists = []
    for arg in args:
        if not isinstance(arg, list):
            arg = [arg]
        lists.append(arg)
    result = list(chain(*lists))
    return result

timex_res = [
    #*[ join_re(period, postfix) for period in periods ]
    *[ join_re(period, postfix) for period, postfix in product(periods, postfixes) ]
    ,*[ join_re(prefix, period) for period in periods ]
    ,*[ join_re(prefix, prefix, period) for period in periods ]
    ,*[ join_re(prefix, abs_time) for abs_time in abs_times ]
    ,*[ join_re(prefix, prefix, abs_time) for abs_time in abs_times ]
    ,*abs_times
    ,*periods
    ,*others
]


for_ = re.compile(r'for$')
the = re.compile(r'the$')

anti_timex_res = [
    {'r':[for_, the], 'p':[1,1], 'n':[0,1]}
]

for abs_time in abs_times:
    anti_timex_res.append({
        'r': [for_, *abs_time]
        ,'p': [1] * (len(abs_time) + 1)
        ,'n': [0, *([1] * len(abs_time))]
    })

from_ = re.compile(r'from$')
to = re.compile(r'to$')

#timex_res.extend([
#    join_re(from_, a, to, b) for a, b in product(timex_res, timex_res)
#])

if False: # False data pattern matcher
    timex_res = [
        join_re(
            re.compile('timex')
            ,re.compile('at')
        )
    ]
    anti_timex_res = [
        {'r':join_re(
            re.compile('timex')
            ,re.compile('at')
        )
        ,'p':[1,1],'n':[1,0]}
    ]

def get_weights(data):
    n = 0
    n_ev = 0
    for s in tempeval2.sentences(data):
        for t in s:
            if 'timexes' not in t:
                n += 1
            else:
                n_ev += 1
    return n/n_ev

def evaluate(data, timex_res, anti_timex_res, quiet=False):
    correct = 0
    total = 0
    w = get_weights(data)
    totals = {'t':{'n':0,'p':0},'f':{'n':0,'p':0}}
    n_sent = 0
    n_corr = 0
    for s in tempeval2.sentences(data):
        n_sent += 1
        text = ' '.join(t['token'] for t in s)
        prediction = find_timexes(s, timex_res, anti_timex_res, quiet=quiet)
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
    sacc = n_corr/n_sent
    score = correct/total
    p = totals['t']['p']/(totals['t']['p'] + totals['f']['p'])
    r = totals['t']['p']/(totals['t']['p'] + totals['f']['n'])
    f = 2*p*r/(p+r)
    if not quiet:
        print("SAcc %.2f (%d/%d)" % (sacc * 100, n_corr, n_sent))
        print("WAcc %.2f" % (score * 100))
        print("Prec %.2f" % (p * 100))
        print("Rec  %.2f" % (r * 100))
        print("F    %.2f" % (f * 100))
    return {
        'p': p
        ,'r': r
        ,'f': f
    }

def run_experiment(k):
    results = evaluate(data_train, timex_res, anti_timex_res, quiet=True)
    return [{
        'name':'Regular Expression parser'
        ,'cross':None
        ,**results
    }]

if __name__ == '__main__':
    baseline_f = evaluate(data_train, timex_res, anti_timex_res)
