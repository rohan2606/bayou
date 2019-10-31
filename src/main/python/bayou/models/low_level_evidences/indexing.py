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
import bayou.models.low_level_evidences.infer
from bayou.models.low_level_evidences.utils import read_config, normalize_log_probs, find_my_rank, rank_statistic, ListToFormattedString
from bayou.models.low_level_evidences.data_reader import Reader


#%%


def index(clargs):
    #set clargs.continue_from = True while testing, it continues from old saved config
    clargs.continue_from = None

    model = bayou.models.low_level_evidences.infer.BayesianPredictor

    # load the saved config
    with open(os.path.join(clargs.save, 'config.json')) as f:
        config = read_config(json.load(f), chars_vocab=True)
    config.batch_size = 500

    reader = Reader(clargs, config, infer=True, dataIsThere=True)

	# Placeholders for tf data
    

    # Placeholders for tf data
    nodes_placeholder = tf.placeholder(reader.nodes.dtype, reader.nodes.shape)
    edges_placeholder = tf.placeholder(reader.edges.dtype, reader.edges.shape)
    targets_placeholder = tf.placeholder(reader.targets.dtype, reader.targets.shape)
    evidence_placeholder = [tf.placeholder(input.dtype, input.shape) for input in reader.inputs[:-1]]
    surr_evidence_placeholder = [tf.placeholder(surr_input.dtype, surr_input.shape) for surr_input in reader.inputs[-1][:-1]]
    surr_evidence_fps_placeholder = [tf.placeholder(surr_fp_input_var.dtype, surr_fp_input_var.shape) for surr_fp_input_var in reader.inputs[-1][-1]]

    feed_dict={fp: f for fp, f in zip(evidence_placeholder, reader.inputs[:-1])}
    feed_dict.update({fp: f for fp, f in zip(surr_evidence_placeholder, reader.inputs[-1][:-1])})
    feed_dict.update({fp: f for fp, f in zip(surr_evidence_fps_placeholder, reader.inputs[-1][-1])})


    feed_dict.update({nodes_placeholder: reader.nodes})
    feed_dict.update({edges_placeholder: reader.edges})
    feed_dict.update({targets_placeholder: reader.targets})

    dataset = tf.data.Dataset.from_tensor_slices(( nodes_placeholder, edges_placeholder, targets_placeholder, *evidence_placeholder, *surr_evidence_placeholder, *surr_evidence_fps_placeholder))
    batched_dataset = dataset.batch(config.batch_size)
    iterator = batched_dataset.make_initializable_iterator()

    jsp = reader.js_programs



    with tf.Session() as sess:
        predictor = model(clargs.save, sess, config, iterator) # goes to infer.BayesianPredictor
        sess.run(iterator.initializer, feed_dict=feed_dict)
        infer_vars = {}


        allEvSigmas = predictor.get_ev_sigma()
        print(allEvSigmas)


        programs = []
        step=200
        start = 18*step
        end = 100*step
        end = min(end , config.num_batches)
        print(f'Running from {start} to {end}')
        for j in range(end):
            if j < start:
                 new_batch = iterator.get_next()
                 continue
            prob_Y, a1,b1, a2, b2 = predictor.get_all_params_inago()
            for i in range(config.batch_size):
                infer_vars = jsp[i]
                prog_json = deepcopy(jsp[   j * config.batch_size + i   ])
                prog_json['a2'] =   "%.3f" % a2[i].item()
                prog_json['b2'] =   [ "%.3f" % val.item() for val in b2[i]]
                #prog_json['a1'] =   "%.3f" % a1[i].item()
                #prog_json['b1'] =   [ "%.3f" % val.item() for val in b1[i]]
                prog_json['ProbY'] = "%.3f" % prob_Y[i].item()
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
    clargs = parser.parse_args(['--save', 'save_500_ndss_decoder/',
        '/home/ubuntu/DATA-newSurrounding_methodHeaders_train_v2_train.json'])

    sys.setrecursionlimit(clargs.python_recursion_limit)
    index(clargs)
