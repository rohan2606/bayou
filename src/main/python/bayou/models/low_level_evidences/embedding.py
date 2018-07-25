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


#import bayou.models.core.infer
import bayou.models.low_level_evidences.infer
from bayou.models.low_level_evidences.data_reader import Reader
from tensorflow.contrib.tensorboard.plugins import projector
from bayou.models.low_level_evidences.utils import read_config, get_var_list


PATH = os.getcwd()
LOG_DIR = PATH + '/embedding'
# LOG_DIR = PATH + '\\embedding'

_classes = ['java.util', 'android.app', 'android.view', 'android.widget', 'java.io', 'javax.xml', 'java.net', \
'android.graphics', 'android.content', 'android.webkit']

HELP="""\
Will add later
"""

#%%

def embedding(input_tensor, labels, sess):
   metadata = os.path.join(LOG_DIR, 'metadata.tsv')

   Embedding = tf.Variable( input_tensor , name='Embedding')


   with open(metadata, 'w') as metadata_file:
       for row in range(input_tensor.shape[0]):
           c=labels[row]
           metadata_file.write('{}\n'.format(c))

   saver = tf.train.Saver([Embedding])

   sess.run(Embedding.initializer)
   saver.save(sess, os.path.join(LOG_DIR, 'Embedding.ckpt'))

   config = projector.ProjectorConfig()
   # One can add multiple embeddings.
   embedding = config.embeddings.add()
   embedding.tensor_name = Embedding.name
   # Link this tensor to its metadata file (e.g. labels).
   embedding.metadata_path = metadata
   # Saves a config file that TensorBoard will read during startup.
   projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)




def get_class_label_new(y, _rev_dict):
    _labels_batch = []
    for arr in y:
        flag = 0
        _dict = {}
        for val in arr:
            API_call = _rev_dict[val]
            starter = ".".join(API_call.split(".")[:2])
            if '$NOT$' in starter:
                starter = starter[5:]
            if starter not in _dict:
                _dict[starter] = 1
            else:
                _dict[starter] += 1

        max_val = -1
        max_key = 'None'
        for key in _dict.keys():
            if key == 'STOP' or key == 'DBranch' or key == 'DLoop':
                continue
            if _dict[key] > max_val:
                max_val = _dict[key]
                max_key = key
        _labels_batch.append(max_key)
    return _labels_batch

#%%
def embed(clargs):
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
    reader = Reader(clargs, config)


    with tf.Session() as sess:
        predictor = model(clargs.save, sess, config, bayou_mode = True) # goes to infer.BayesianPredictor
        reader.reset_batches()

        _psi_encoders = []
        _labels = []
        _rev_dict = {v: k for k, v in config.decoder.vocab.items()}

        for b in range(config.num_batches):
            # setup the feed dict
            prog_ids, ev_data, n, e, y = reader.next_batch()
            feed = {predictor.model.targets: y}
            for j, ev in enumerate(config.evidence):
                feed[predictor.model.encoder.inputs[j].name] = ev_data[j]
            for j in range(config.decoder.max_ast_depth):
                feed[predictor.model.decoder.nodes[j].name] = n[j]
                feed[predictor.model.decoder.edges[j].name] = e[j]
            # Feeding value into reverse encoder
                feed[predictor.model.reverse_encoder.nodes[j].name] = n[config.decoder.max_ast_depth - 1 - j]
                feed[predictor.model.reverse_encoder.edges[j].name] = e[config.decoder.max_ast_depth - 1 - j]

            _labels.extend(get_class_label_new(y,_rev_dict))
            # run the optimizer
            _psi_encoder \
                = sess.run(predictor.model.psi_encoder, feed)

            _psi_encoders.append(_psi_encoder)

        _psi_encoders_agg = np.concatenate(_psi_encoders, axis = 0)
        embedding(_psi_encoders_agg,_labels, sess)


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
    clargs = parser.parse_args(['--save', 'save',
    #'..\low_level_evidences\save',
    #'..\..\..\..\..\..\data\DATA-training-top.json'])
    '/home/ubuntu/DATA-top.json'])


    sys.setrecursionlimit(clargs.python_recursion_limit)
    embed(clargs)
