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

import bayou.models.non_prob.infer
from bayou.models.non_prob.data_reader import Reader
from bayou.models.low_level_evidences.utils import read_config, normalize_log_probs, find_my_rank, rank_statistic, ListToFormattedString


#%%

def test(clargs):
    encs, rev_encs  = test_get_vals(clargs)


    latent_size, num_progs, batch_size = len(encs[0]), len(encs), 10000
    hit_points = [1,2,5,10,50,100,500,1000,5000,10000]
    hit_counts = np.zeros(len(hit_points))
    for i in range(num_progs):
        distances = []
        for j in range(int(np.ceil(num_progs / batch_size))):
            sid, eid = j * batch_size, min( (j+1) * batch_size , num_progs)
            dist = cosine_similarity(np.array(encs[i:i+1]), np.array(rev_encs[sid:eid]))
            distances += list(dist)
        
        _rank = find_my_rank( distances , i )

        hit_counts, prctg = rank_statistic(_rank, i + 1, hit_counts, hit_points)

        if (((i+1) % 100 == 0) or (i == (num_progs - 1))):
            print('Searched {}/{} (Max Rank {})'
                  'Hit_Points {} :: Percentage Hits {}'.format
                  (i + 1, num_progs, num_progs,
                   ListToFormattedString(hit_points, Type='int'), ListToFormattedString(prctg, Type='float')))
    return


def test_get_vals(clargs):

    infer_vars, config = forward_pass(clargs)

    encs, rev_encs  = [],[]
    for prog_id in sorted(list(infer_vars.keys())):
       encs += [list(infer_vars[prog_id]['psi_enc'])]
       rev_encs += [list(infer_vars[prog_id]['psi_rev_enc'])]

    return encs, rev_encs

def forward_pass(clargs):
    #set clargs.continue_from = True while testing, it continues from old saved config
    clargs.continue_from = True

    with open(os.path.join(clargs.save, 'config.json')) as f:
        model_type = json.load(f)['model']

    if model_type == 'lle':
        model = bayou.models.non_prob.infer.BayesianPredictor
    else:
        raise ValueError('Invalid model type in config: ' + model_type)

    # load the saved config
    with open(os.path.join(clargs.save, 'config.json')) as f:
        config = read_config(json.load(f), chars_vocab=True)

    reader = Reader(clargs, config, infer=True)

    # merged_summary = tf.summary.merge_all()

    # Placeholders for tf data
    nodes_placeholder = tf.placeholder(reader.nodes.dtype, reader.nodes.shape)
    edges_placeholder = tf.placeholder(reader.edges.dtype, reader.edges.shape)

    evidence_placeholder = [tf.placeholder(input.dtype, input.shape) for input in reader.inputs[:-1]]
    surr_evidence_placeholder = [tf.placeholder(surr_input.dtype, surr_input.shape) for surr_input in reader.inputs[-1][:-1]]
    surr_evidence_fps_placeholder = [tf.placeholder(surr_fp_input_var.dtype, surr_fp_input_var.shape) for surr_fp_input_var in reader.inputs[-1][-1]]


    neg_evidence_placeholder = [tf.placeholder(input.dtype, input.shape) for input in reader.inputs[:-1]]
    neg_surr_evidence_placeholder = [tf.placeholder(surr_input.dtype, surr_input.shape) for surr_input in reader.inputs[-1][:-1]]
    neg_surr_evidence_fps_placeholder = [tf.placeholder(surr_fp_input_var.dtype, surr_fp_input_var.shape) for surr_fp_input_var in reader.inputs[-1][-1]]

    feed_dict={fp: f for fp, f in zip(evidence_placeholder, reader.inputs[:-1])}
    feed_dict.update({fp: f for fp, f in zip(surr_evidence_placeholder, reader.inputs[-1][:-1])})
    feed_dict.update({fp: f for fp, f in zip(surr_evidence_fps_placeholder, reader.inputs[-1][-1])})

    feed_dict_neg={fp: f for fp, f in zip(neg_evidence_placeholder, reader.inputs_negative[:-1])}
    feed_dict_neg.update({fp: f for fp, f in zip(neg_surr_evidence_placeholder, reader.inputs_negative[-1][:-1])})
    feed_dict_neg.update({fp: f for fp, f in zip(neg_surr_evidence_fps_placeholder, reader.inputs_negative[-1][-1])})

    feed_dict.update(feed_dict_neg)
    feed_dict.update({nodes_placeholder: reader.nodes})
    feed_dict.update({edges_placeholder: reader.edges})

    dataset = tf.data.Dataset.from_tensor_slices(( nodes_placeholder, edges_placeholder,  *evidence_placeholder, *surr_evidence_placeholder, *surr_evidence_fps_placeholder, \
                                        *neg_evidence_placeholder, *neg_surr_evidence_placeholder, *neg_surr_evidence_fps_placeholder))
    batched_dataset = dataset.batch(config.batch_size)
    iterator = batched_dataset.make_initializable_iterator()


	# Placeholders for tf data
    jsp = reader.js_programs
    with tf.Session() as sess:
        predictor = model(clargs.save, sess, config, iterator) # goes to infer.BayesianPredictor
        # testing
        sess.run(iterator.initializer, feed_dict=feed_dict)
        infer_vars = {}


        for j in range(config.num_batches):
            psi_enc, psi_neg, psi_rev_enc = predictor.get_all_latent_vectors()
            for i in range(config.batch_size):
                prog_id = j * config.batch_size + i
                infer_vars[prog_id] = {}
                infer_vars[prog_id]['psi_enc'] = psi_enc[i].round(decimals=2)
                infer_vars[prog_id]['psi_rev_enc'] = psi_rev_enc[i].round(decimals=2)


            if (j+1) % 1000 == 0:
                print('Completed Processing {}/{} batches'.format(j+1, config.num_batches))



    return infer_vars, config


def cosine_similarity(a, b):
   norm_denom_a = np.linalg.norm(a,axis=1)
   norm_a = a/norm_denom_a[:,None]

   norm_denom_b = np.linalg.norm(b, axis=1)
   norm_b = b/norm_denom_b[:, None]

   sim = np.dot(norm_a, np.transpose(norm_b))
   return sim[0]



def euclidean_distance(a,b):
    return np.sum(np.square(b-a[0]) , axis=1)

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
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
    clargs = parser.parse_args(['--save', 'save_no_max'])

    sys.setrecursionlimit(clargs.python_recursion_limit)
    test(clargs)
