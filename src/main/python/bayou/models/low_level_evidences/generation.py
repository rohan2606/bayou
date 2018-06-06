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
import os
import sys
import json
import textwrap
import random
import time
#import bayou.models.core.infer
import bayou.models.low_level_evidences.infer
from bayou.models.low_level_evidences.utils import read_config, normalize_log_probs, find_my_rank, rank_statistic, ListToFormattedString
from bayou.models.low_level_evidences.data_reader import Reader
from bayou.models.low_level_evidences.test import get_c_minus_cstar
from bayou.models.low_level_evidences.utils import plot_probs, find_top_rank_ids, normalize_log_probs, find_my_rank


File_Name = 'Search_Data_Basic'

HELP = """
 """
#%%


def get_a1b1(clargs):
    clargs.continue_from = True
    # load the saved config
    with open(os.path.join(clargs.save, 'config.json')) as f:
        config = read_config(json.load(f), chars_vocab=True)

    random.seed(3)
    batch_id = random.randint(0,config.batch_size-1)
    print(batch_id)

    with open(os.path.join(clargs.save, 'config.json')) as f:
        model_type = json.load(f)['model']

    if model_type == 'lle':
        model = bayou.models.low_level_evidences.infer.BayesianPredictor
    else:
        raise ValueError('Invalid model type in config: ' + model_type)

    reader = Reader(clargs, config)
    reader.reset_batches()

    with tf.Session() as sess:
        config.batch_size = 1
        predictor = model(clargs.save, sess, config, bayou_mode = False) # goes to infer.BayesianPredictor
        prog_id, ev_data, _, _, y = reader.next_batch()

        prog_id = prog_id[batch_id]
        ev_data = [ev[batch_id:batch_id+1] for ev in ev_data]
        y = y[batch_id]

        feed = {}
        for j, ev in enumerate(config.evidence):
            feed[predictor.model.encoder.inputs[j].name] = ev_data[j]

        a1, b1 = sess.run([predictor.model.EncA, predictor.model.EncB], feed)

    for i, ev in enumerate(config.evidence):
        ev.print_ev(ev_data[i])


    inv_map = {v: k for k, v in config.decoder.vocab.items()}
    print('\nThe original sequence might be this (BEWARE :: However can also be seq from other program part)')
    for call in y:
        string = inv_map[call]
        if string == 'STOP':
            print('' , end='')
        else:
            print(string , end=',')
    print()
    return a1,b1, prog_id, config,

def test_get_vals(clargs):
    a2s = np.load(File_Name  + '/a2s.npy')
    b2s = np.load(File_Name  + '/b2s.npy')
    prob_Ys = np.load(File_Name  + '/prob_Ys.npy')
    Ys = np.load(File_Name  + '/Ys.npy')
    return a2s, b2s, prob_Ys, Ys

def test(clargs):

    [a1, b1, prog_id, config] = get_a1b1(clargs)
    [a2s, b2s, prob_Ys, Ys] = test_get_vals(clargs)

    latent_size, num_progs, batch_size = len(b2s[0]), len(a2s), 1000

    prob_Y_Xs = []
    for j in range(int(np.ceil(num_progs / batch_size))):
        sid, eid = j * batch_size, min( (j+1) * batch_size , num_progs)
        prob_Y_X = get_c_minus_cstar(np.array(a1[0]), np.array(b1[0]),\
                                np.array(a2s[sid:eid]), np.array(b2s[sid:eid]), np.array(prob_Ys[sid:eid]), latent_size)
        prob_Y_Xs += list(prob_Y_X)


    prob_Y_Xs = normalize_log_probs(prob_Y_Xs)
    rank_ids, sorted_probs = find_top_rank_ids( prob_Y_Xs, cutoff = 25)
    #pred_rank = find_my_rank(prob_Y_Xs, prog_id)
    inv_map = {v: k for k, v in config.decoder.vocab.items()}

    #print('\nPredicted Rank is {}'.format(pred_rank)) # not possible as JSON is unordered object
    print()
    for rank, jid in enumerate(rank_ids):
        print('Rank :: {} , LogProb :: {}'.format(rank + 1, sorted_probs[rank]))
        for i, prog_trace in enumerate(Ys[jid]):
            for call in prog_trace:
                string = inv_map[call]
                if string == 'STOP':
                    print('' , end='')
                else:
                    print(string , end=',')
            print()
        print()
    print()
    plot_probs(sorted_probs)
    plot_probs(sorted_probs[:100], fig_name ="rankedProbtop100.pdf")
    plot_probs(sorted_probs[:10], fig_name ="rankedProbtop10.pdf")

    return


#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent(HELP))
    parser.add_argument('input_file', type=str, nargs=1,
                        help='input data file')
    parser.add_argument('--python_recursion_limit', type=int, default=10000,
                        help='set recursion limit for the Python interpreter')
    parser.add_argument('--save', type=str, required=True,
                        help='checkpoint model during training here')
    parser.add_argument('--evidence', type=str, default='all',
                        choices=['apicalls', 'types', 'keywords', 'all'],
                        help='use only this evidence for inference queries')
    parser.add_argument('--output_file', type=str, default=None,
                        help='output file to print probabilities')

    #clargs = parser.parse_args()
    parseJSON = False
    if parseJSON:
        clargs = parser.parse_args(['--save', 'save', 'generation/query.json'])
    else:
        clargs = parser.parse_args(['--save', 'save', '/home/rm38/Research/Bayou_Code_Search/Corpus/DATA-training-expanded-biased-TOP.json'])



    sys.setrecursionlimit(clargs.python_recursion_limit)
    test(clargs)
