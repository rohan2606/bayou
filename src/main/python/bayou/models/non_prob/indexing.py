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
import simplejson as json
import textwrap
import gc
from copy import deepcopy

#import bayou.models.core.infer
import bayou.models.non_prob.infer
from bayou.models.low_level_evidences.utils import  normalize_log_probs, find_my_rank, rank_statistic, ListToFormattedString
from bayou.models.non_prob.utils import read_config
from bayou.models.non_prob.data_reader import Reader


#%%


def index(clargs):
    #set clargs.continue_from = True while testing, it continues from old saved config
    clargs.continue_from = None

    model = bayou.models.non_prob.infer.BayesianPredictor

    # load the saved config
    with open(os.path.join(clargs.save, 'config.json')) as f:
        config = read_config(json.load(f), chars_vocab=True)
    config.batch_size = 50

    reader = Reader(clargs, config, infer=True, dataIsThere=True)

    # Placeholders for tf data
    nodes_placeholder = tf.placeholder(reader.nodes.dtype, reader.nodes.shape)
    edges_placeholder = tf.placeholder(reader.edges.dtype, reader.edges.shape)

    evidence_placeholder = [tf.placeholder(input.dtype, input.shape) for input in reader.inputs]


    neg_evidence_placeholder = [tf.placeholder(input.dtype, input.shape) for input in reader.inputs]

    feed_dict={fp: f for fp, f in zip(evidence_placeholder, reader.inputs)}

    feed_dict_neg={fp: f for fp, f in zip(neg_evidence_placeholder, reader.inputs_negative)}

    feed_dict.update(feed_dict_neg)
    feed_dict.update({nodes_placeholder: reader.nodes})
    feed_dict.update({edges_placeholder: reader.edges})

    dataset = tf.data.Dataset.from_tensor_slices(( nodes_placeholder, edges_placeholder,  *evidence_placeholder, *neg_evidence_placeholder))
    batched_dataset = dataset.batch(config.batch_size)
    iterator = batched_dataset.make_initializable_iterator()


    jsp = reader.js_programs


    with tf.Session() as sess:
        predictor = model(clargs.save, sess, config, iterator) # goes to infer.BayesianPredictor
        sess.run(iterator.initializer, feed_dict=feed_dict)
        infer_vars = {}

        programs = []
        step=200
        start = 0*step
        end = 18*step
        end = min(end , config.num_batches)
        #print(f'Running from {start} to {end}')
        for j in range(end):
            if j < start:
                 new_batch = iterator.get_next()
                 continue
            enc_psi, enc_neg_psi, rev_enc_psi = predictor.get_all_latent_vectors()
            for i in range(config.batch_size):
                infer_vars = jsp[i]
                prog_json = deepcopy(jsp[   j * config.batch_size + i   ])
                prog_json['prog_psi'] =  [ "%.3f" % val.item() for val in enc_psi[i]]
                prog_json['prog_neg_psi'] =  [ "%.3f" % val.item() for val in enc_neg_psi[i]]
                prog_json['prog_psi_rev'] =  [ "%.3f" % val.item() for val in rev_enc_psi[i]]
                programs.append(prog_json)

            if (j+1) % step == 0 or (j+1) == config.num_batches:
                k = (j+1)//step
                fileName = "Program_output_" + str(k) + ".json"
                print('\nWriting to {}...'.format(fileName), end='')
                with open(fileName, 'w') as f:
                     json.dump({'programs': programs}, fp=f, indent=2)

                for item in programs:
                    del item
                del programs
                gc.collect()
                programs = []



    print('Batch Processing Completed')

    return infer_vars, config




#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
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
    clargs = parser.parse_args(['--save', 'save_no_bound',
        '/home/ubuntu/DATA-newSurrounding_methodHeaders_train_v2_train.json'])

    sys.setrecursionlimit(clargs.python_recursion_limit)
    index(clargs)
