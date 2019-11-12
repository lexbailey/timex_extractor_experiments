def red(s):
    return '\u001b[31m%s\u001b[0m'%s

def green(s):
    return '\u001b[32m%s\u001b[0m'%s

def find_timexes_nohyphen(s, timex_res, anti_timex_res, quiet=False):
    result = []
    seen = []
    for t in s:
        word = t['token'].lower()
        seen.append(word)
        result.append(0)
        for r in timex_res:
            if len(r) <= len(seen):
                all_match = True
                for i, r2 in enumerate(r):
                    this_word = seen[len(seen)-len(r)+i]
                    if not r2.match(this_word):
                        all_match = False
                        break
                if all_match:
                    result[-1] = 1
                    for i in range(1,len(r)):
                        result[-1-i] = 1
    seen = []
    result_before = result.copy()
    for t in s:
        word = t['token'].lower()
        seen.append(word)
        for a in anti_timex_res:
            if len(a['r']) <= len(seen):
                all_match = True
                for i, (r, p, n) in enumerate(zip(a['r'], a['p'], a['n'])):
                    this_word = seen[len(seen)-len(a['r'])+i]
                    this_p = result[len(seen)-len(a['r'])+i]
                    if this_p != p or not r.match(this_word):
                        all_match = False
                        break
                if all_match:
                    for i, n in enumerate(reversed(a['n'])):
                        result[len(seen)-1-i] = n
    all_correct = True
    for t, g in zip(s, result):
        if g != (1 if 'timexes' in t else 0):
            all_correct = False
    if not all_correct:
        true = ' '.join([
            t['token'] if 'timexes' not in t else green(t['token']) for t in s
        ])
        guess = ' '.join([
            t['token'] if g == 0 else red(t['token']) for t, g in zip(s, result)
        ])
        if not quiet:
            print()
            print(true)
            print(guess)

    return result

def find_timexes(s, timex_res, anti_timex_res, quiet=False):
    true = ' '.join([
        t['token'] for t in s
    ])
    new_s = []
    next_group = 0
    for t in s:
        parts = t['token'].split('-')
        for part in parts:
            new = t.copy()
            new['token'] = part
            new['group'] = next_group
            new_s.append(new)
        next_group += 1
    result = find_timexes_nohyphen(new_s, timex_res, anti_timex_res, quiet)
    if len(new_s) > len(s):
        new_result = []
        last_group = -1
        for t, res in zip(new_s, result):
            if t['group'] != last_group:
                new_result.append(res)
            last_group = t['group']
        result = new_result
    return result
