# Copyright 2017 Rice University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
import sys
import json
import textwrap

from data_reader import Reader
from utils import read_config, dump_config, normalize_log_probs, find_my_rank, rank_statistic, ListToFormattedString

HELP="""
"""
#%%



def train(clargs):
    config_file = clargs.config if clargs.continue_from is None \
                                else os.path.join(clargs.continue_from, 'config.json')

    with open(config_file) as f:
        config = read_config(json.load(f), chars_vocab=clargs.continue_from)

    reader = Reader(clargs, config)
    file_ptr_dict = np.load(os.path.join(clargs.save, 'file_ptr.pkl'))

    jsconfig = dump_config(config)
    # print(clargs)
    # print(json.dumps(jsconfig, indent=2))

    with open(os.path.join(clargs.save, 'config.json'), 'w') as f:
        json.dump(jsconfig, fp=f, indent=2)


    # print(config.num_batches)
    evidence_list = [dict() for i in range(4)]
    reader.reset_batches()
    for b in range(config.num_batches):
        prog_ids, ev_data, _, _, _, _ = reader.next_batch()
        for j, prog_id in enumerate(prog_ids):
            for i in range(len(config.evidence)):
                if i != 3:
                    evidence_list[i][prog_id] = ev_data[i][j][0]
                else:
                    # print (ev_data[i][j])
                    evidence_list[i][prog_id] = ev_data[i][j]


    all_progs = evidence_list[0].keys()

    # print(evidence_list[3])

    hit_points = [2,5,10,50,100,500,1000,5000,10000]
    hit_counts = np.zeros(len(hit_points))
    num_progs = len(all_progs)
    prob = np.zeros(len(all_progs))
    for i, q in enumerate(all_progs): #query prog
        q_ev = [evidence_list[c][q] for c in range(len(config.evidence))]
        for p in all_progs:
            prob[p] = 0.0
            found_file = parse_filePtr(file_ptr_dict[p])
            file_val = open(found_file, errors = 'replace').read().lower()
            for j, ev in enumerate(config.evidence):
                prob[p] += ev.count_occurence(q_ev[j], found_file)
        _rank = find_my_rank( prob , i )

        hit_counts, prctg = rank_statistic(_rank, i + 1, hit_counts, hit_points)

        if (((i+1) % 1 == 0) or (i == (num_progs - 1))):
            print('Searched {}/{} (Max Rank {})'
                  'Hit_Points {} :: Percentage Hits {}'.format
                  (i + 1, num_progs, num_progs,
                   ListToFormattedString(hit_points, Type='int'), ListToFormattedString(prctg, Type='float')))



def parse_filePtr(filePtr):
    return '/'.join(['/home/ec2-user','Corpus', 'java_projects'] + filePtr.split('/')[1:])

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent(HELP))
    parser.add_argument('input_file', type=str, nargs=1,
                        help='input data file')
    parser.add_argument('--python_recursion_limit', type=int, default=10000,
                        help='set recursion limit for the Python interpreter')
    parser.add_argument('--save', type=str, default='save',
                        help='checkpoint model during training here')
    parser.add_argument('--config', type=str, default=None,
                        help='config file (see description above for help)')
    parser.add_argument('--continue_from', type=str, default=None,
                        help='ignore config options and continue training model checkpointed here')
    #clargs = parser.parse_args()
    clargs = parser.parse_args(
     # ['--continue_from', 'save',
     ['--config','config.json',
     # '..\..\..\..\..\..\data\DATA-training-top.json'])
    # '/home/rm38/Research/Bayou_Code_Search/Corpus/DATA-training-expanded-biased-TOP.json'])
     '/home/ec2-user/Corpus/DATA-training-expanded-biased-TOP.json'])
    sys.setrecursionlimit(clargs.python_recursion_limit)
    if clargs.config and clargs.continue_from:
        parser.error('Do not provide --config if you are continuing from checkpointed model')
    if not clargs.config and not clargs.continue_from:
        parser.error('Provide at least one option: --config or --continue_from')
    train(clargs)
