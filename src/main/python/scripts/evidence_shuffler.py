import json
from random import shuffle
import sys
sys.setrecursionlimit(10000)

print('Loading data file...')
with open('/home/ubuntu/DATA-training-expanded-biased.json') as f:
    js = json.load(f)
print('done')

shuffle(js['programs'])

with open('data.json', 'w') as outfile:
    json.dump(js, outfile, indent=1)
