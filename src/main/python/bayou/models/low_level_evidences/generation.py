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
from datetime import datetime
import time

#import bayou.models.core.infer
import bayou.models.low_level_evidences.infer
from bayou.models.low_level_evidences.utils import read_config, normalize_log_probs, rank_statistic, ListToFormattedString
from bayou.models.low_level_evidences.data_reader import Reader
from bayou.models.low_level_evidences.test import get_c_minus_cstar
from bayou.models.low_level_evidences.utils import plot_probs, normalize_log_probs


File_Name = 'Search_Data_Basic'

HELP = """
 """
#%%

time_id = datetime.now()
file_timestamp = str(time_id)[5:-7].replace(' ',':')

def get_a1b1(clargs, f):
    clargs.continue_from = True

    with open(os.path.join(clargs.save, 'config.json')) as f1:
        config = read_config(json.load(f1), chars_vocab=True)

    with open(os.path.join(clargs.save, 'config.json')) as f1:
        model_type = json.load(f1)['model']

    if model_type == 'lle':
        model = bayou.models.low_level_evidences.infer.BayesianPredictor
    else:
        raise ValueError('Invalid model type in config: ' + model_type)


    file_ptr_dict = np.load(os.path.join(clargs.save, 'file_ptr.pkl'))

    reader = Reader(clargs, config)
    reader.reset_batches()

    random.seed(time_id)
    batch_id = random.randint(0,config.batch_size-1)
    batch_num = random.randint(1,config.num_batches)

    with tf.Session() as sess:
        config.batch_size = 1
        predictor = model(clargs.save, sess, config, bayou_mode = False) # goes to infer.BayesianPredictor
        for b in range(batch_num):
            prog_ids, ev_data, _, _, y = reader.next_batch()

        prog_id = prog_ids[batch_id]
        #f.write('\nOriginal File ptr :: ' + str(parse_filePtr(file_ptr_dict[prog_id])) + '\n\n\n')
        ev_data = [ev[batch_id:batch_id+1] for ev in ev_data]
        y = y[batch_id]

        feed = {}
        for j, ev in enumerate(config.evidence):
            feed[predictor.model.encoder.inputs[j].name] = ev_data[j]

        a1, b1 = sess.run([predictor.model.EncA, predictor.model.EncB], feed)

    for i, ev in enumerate(config.evidence):
        print(ev.name)
        ev.f_write(ev_data[i], f)

    return a1,b1, prog_id, config,


def test(clargs, f):
    file_ptr_dict = np.load(os.path.join(clargs.save, 'file_ptr.pkl'))

    [a1, b1, prog_id, config] = get_a1b1(clargs, f)
    [a2s, b2s, prob_Ys, Ys] = test_get_vals(clargs)

    latent_size, num_progs, batch_size = len(b2s[0]), len(a2s), 1000

    prob_Y_Xs = []
    for j in range(int(np.ceil(num_progs / batch_size))):
        sid, eid = j * batch_size, min( (j+1) * batch_size , num_progs)
        prob_Y_X = get_c_minus_cstar(np.array(a1[0]), np.array(b1[0]),\
                                np.array(a2s[sid:eid]), np.array(b2s[sid:eid]), np.array(prob_Ys[sid:eid]), latent_size)
        prob_Y_Xs += list(prob_Y_X)

    prob_Y_Xs = normalize_log_probs(prob_Y_Xs)

    # rank_ids, sorted_probs = find_top_rank_ids( prob_Y_Xs, cutoff = 10)
    # pred_rank = find_my_rank(prob_Y_Xs,  prog_id )
    # f.write('\nPredicted Rank is {}'.format(pred_rank + 1))

    inv_map = {v: k for k, v in config.decoder.vocab.items()}


    for rank, jid in enumerate(rank_ids):
        f.write('\n\n\nRank :: {} , LogProb :: {}\n\n'.format(rank + 1, sorted_probs[rank]))
        found_file = parse_filePtr(file_ptr_dict[jid])
        f.write('File Ptr ::' + found_file)
        f.write('\n\nPaths in the AST::\n')
        call_array = []
        for i, prog_trace in enumerate(Ys[jid]):
            trace_array = []
            for call in prog_trace:
                string = inv_map[call]
                if string != 'STOP':
                    f.write(string + ' , ')
            f.write('\n')
        f.write('\nSource Code :: \n\n\n')
        # TURN THIS ON FOR FULL CODE`
        #f.write(open(found_file).read())

    plot_probs(sorted_probs, fig_name ="rankedProbAll"+ file_timestamp +".pdf")
    plot_probs(sorted_probs[:100], fig_name ="rankedProbtop100" + file_timestamp +".pdf")
    plot_probs(sorted_probs[:10], fig_name ="rankedProbtop10" + file_timestamp +".pdf")

    return

def parse_filePtr(filePtr):
    return '/'.join(['/home/ubuntu','Corpus', 'java_projects'] + filePtr.split('/')[1:])

def test_get_vals(clargs):
    a2s = np.load(File_Name  + '/a2s.npy')
    b2s = np.load(File_Name  + '/b2s.npy')
    prob_Ys = np.load(File_Name  + '/prob_Ys.npy')
    Ys = np.load(File_Name  + '/Ys.npy')
    return a2s, b2s, prob_Ys, Ys

def find_top_rank_ids(arrin, cutoff = 10):
    rank_ids =  (-np.array(arrin)).argsort()
    vals = []
    for rank in rank_ids:
        vals.append(arrin[rank])
    return rank_ids[:cutoff], vals

def find_my_rank(arr, i):
    pivot = arr[i]
    rank = 0
    for val in arr:
        if val > pivot:
            rank += 1
    return rank


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
        clargs = parser.parse_args(['--save', 'save1', 'generation/query.json'])
    else:
        clargs = parser.parse_args(['--save', 'save1', '/home/ubuntu/DATA-retry.json'])
#'/home/ubuntu/Corpus/DATA-training-expanded-biased-TOP.json'])


    f = open("./generation/ResultGeneration"+ file_timestamp + ".txt", "w")
    sys.setrecursionlimit(clargs.python_recursion_limit)
    test(clargs, f)
    f.close()
