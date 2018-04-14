import json
import os

BPTT = 35
DATA_DIR = 'data'

def process_ptb(in_filename, out_filename):
    with open(f'{in_filename}', 'r') as r:
        data = r.read()
        data = data.replace('\n', '<eos>')
        data = data.split()

    examples = []

    for i, _ in enumerate(data[:-BPTT-1]):
        examples.append({'words':data[i:i+BPTT], 'target':data[i+BPTT]})

    with open(f'{out_filename}', 'w') as w:
        for example in examples:
            json.dump(example, w)
            w.write('\n')

process_ptb(os.path.join(DATA_DIR, 'ptb.test.txt'), os.path.join(DATA_DIR, 'ptb.test.jsonl'))
process_ptb(os.path.join(DATA_DIR, 'ptb.valid.txt'), os.path.join(DATA_DIR, 'ptb.valid.jsonl'))
process_ptb(os.path.join(DATA_DIR, 'ptb.train.txt'), os.path.join(DATA_DIR, 'ptb.train.jsonl'))
