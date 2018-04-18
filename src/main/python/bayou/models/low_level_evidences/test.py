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

import time
#import bayou.models.core.infer
import bayou.models.low_level_evidences.infer
from bayou.models.low_level_evidences.utils import read_config, normalize_log_probs, find_my_rank, rank_statistic, ListToFormattedString
from bayou.models.low_level_evidences.data_reader import Reader

HELP = """\
May God help you son/daughter!
"""
#%%

def test(clargs):
    #set clargs.continue_from = True which ignores config options and starts
    #training
    clargs.continue_from = True

    with open(os.path.join(clargs.save, 'config.json')) as f:
        model_type = json.load(f)['model']

    if model_type == 'lle':
        model = bayou.models.low_level_evidences.infer.BayesianPredictor
    else:
        raise ValueError('Invalid model type in config: ' + model_type)

    # load the saved config
    with open(os.path.join(clargs.save, 'config.json')) as f:
        config = read_config(json.load(f), chars_vocab=True)
    reader = Reader(clargs, config, test_mode=True)

    with tf.Session() as sess:
        predictor = model(clargs.save, sess, config, bayou_mode = False) # goes to infer.BayesianPredictor

        # testing
        reader.reset_batches()
        prob_Y, a1s, b1s, a2s, b2s = [], [], [], [], []
        for i in range(config.num_batches):
            ev_data, n, e, y = reader.next_batch()
            prob_Y.append(predictor.get_lnProbY(ev_data, n, e, y))
            a1, b1 = predictor.get_encoder_ab(ev_data)
            a2, b2 = predictor.get_rev_encoder_ab(n,e, ev_data)
            a1s.append(a1), b1s.append(b1)
            a2s.append(a2), b2s.append(b2)
            if i % 100 == 0:
                print('Completed Processing {}/{} batches'.format
                (i+1, config.num_batches))


        a1s,b1s,a2s,b2s = np.concatenate(a1s, axis=0), np.concatenate(b1s, axis=0), \
                            np.concatenate(a2s, axis=0) , np.concatenate(b2s, axis=0)
        prob_Y = normalize_log_probs(np.concatenate(prob_Y, axis=0))

        hit_points = [2,5,10,50,100,500]
        hit_counts = np.zeros(len(hit_points))
        for i in range(config.num_batches * config.batch_size):
            prob_Y_X = []
            for j in range(config.num_batches):
                sid = j * config.batch_size
                eid = (j+1) * config.batch_size
                prob_Y_X_i = predictor.get_c_minus_cstar(a1s[i], b1s[i], a2s[sid:eid], b2s[sid:eid], prob_Y[sid:eid])
                prob_Y_X.extend(prob_Y_X_i)

            assert(len(prob_Y_X) == config.num_batches * config.batch_size)
            _rank = find_my_rank(prob_Y_X, i)
            hit_counts, prctg = rank_statistic(_rank, i + 1, hit_counts, hit_points)

            if (i+1) % 100 == 0:
                print('Searched {}/{} (Max Rank {})'
                      'Hit_Points {} :: Percentage Hits {}'.format
                      (i + 1, config.num_batches*config.batch_size, config.num_batches*config.batch_size,
                       ListToFormattedString(hit_points, Type='int'), ListToFormattedString(prctg, Type='float')))


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
    clargs = parser.parse_args(['--save', 'save2',
    '/home/ubuntu/bayou/data/DATA-training.json'])
    #'..\..\..\..\..\..\data\DATA-training.json'])
#    '/home/rm38/Research/Bayou_Code_Search/bayou/data/DATA-training.json'])


    sys.setrecursionlimit(clargs.python_recursion_limit)
    test(clargs)
