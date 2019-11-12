from StructInflate import Inflator
from os import path
import sys

def parse(folder, enc='utf-8'):
    file_names = {
        'base': 'base-segmentation.tab',
        'event_ext': 'event-extents.tab',
        'event_attr': 'event-attributes.tab',
        'timex_ext': 'timex-extents.tab',
        'timex_attr': 'timex-attributes.tab',
    }
    files = {
        key: open(path.join(folder, filename), encoding=enc) for key, filename in file_names.items()
    }
    inf = Inflator('''
        dict
        >list
        >list
        >dict(
            token=prim,
            events=dict
                  >list
                  >dict(
                      aspect=prim,
                      modality=prim,
                      polarity=prim,
                      tense=prim
                  ),
            timexes=dict
                   >list
                   >dict(
                       type=prim,
                       value=prim
                   )
        )
    ''')
    for line in files['base']:
        fid, sid, tid, token = line.strip().split('\t')
        inf.add([fid, sid, tid, 'token'], token)
    for line in files['event_ext']:
        fid, sid, tid, event, eid, iid = line.strip().split('\t')
        inf.add([fid, sid, tid, 'events', eid, int(iid)-1], inf.placeholder)
    for line in files['event_attr']:
        try:
            fid, sid, tid, event, eid, iid, attr, value = line.strip().split('\t')
            inf.add([fid, sid, tid, 'events', eid, int(iid)-1, attr], value)
        except ValueError:
            #print("Warning, failed to load line from data set: %s" % line, file=sys.stderr)
            pass
    for line in files['timex_ext']:
        fid, sid, tid, event, teid, iid = line.strip().split('\t')
        inf.add([fid, sid, tid, 'timexes', teid, int(iid)-1], inf.placeholder)
    for line in files['timex_attr']:
        try:
            fid, sid, tid, event, teid, iid, attr, value = line.strip().split('\t')
            inf.add([fid, sid, tid, 'timexes', teid, int(iid)-1, attr], value)
        except ValueError:
            print("Warning, failed to load line from data set: %s" % line, file=sys.stderr)

    for f in files.values():
        f.close()
    return inf.data

def sentences(data):
    for k in data:
        for s in data[k]:
            yield s

def reconstruct_sentence(data, fid, sid, tokens=False):
    s = data[fid][sid]
    parts = []
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
    chinese_data = parse('./data/tempeval-training-2/chinese/data', enc='GB18030')
    english_data = parse('./data/tempeval-training-2/english/data')
    spanish_data = parse('./data/tempeval-training-2/spanish/data')
    print(reconstruct_sentence(chinese_data, next(iter(chinese_data.keys())), 2))
    print(reconstruct_sentence(english_data, next(iter(english_data.keys())), 2, tokens=True))
    print(reconstruct_sentence(spanish_data, next(iter(spanish_data.keys())), 2))
