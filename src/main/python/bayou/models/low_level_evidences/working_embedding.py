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
from bayou.models.low_level_evidences.utils import read_config, normalize_log_probs
from bayou.models.low_level_evidences.data_reader import Reader

_classes = ['java.util', 'android.app', 'android.view', 'android.widget', 'java.io', 'javax.xml', 'java.net', \
'android.graphics', 'android.content', 'android.webkit']


HELP = """\
Config options should be given as a JSON file (see config.json for example):
{                                         |
    "model": "lle"                        | The implementation id of this model (do not change)
    "latent_size": 32,                    | Latent dimensionality
    "batch_size": 50,                     | Minibatch size
    "num_epochs": 100,                    | Number of training epochs
    "learning_rate": 0.02,                | Learning rate
    "print_step": 1,                      | Print training output every given steps
    "alpha": 1e-05,                       | Hyper-param associated with KL-divergence loss
    "beta": 1e-05,                        | Hyper-param associated with evidence loss
    "evidence": [                         | Provide each evidence type in this list
        {                                 |
            "name": "apicalls",           | Name of evidence ("apicalls")
            "units": 64,                  | Size of the encoder hidden state
            "num_layers": 3               | Number of densely connected layers
            "tile": 1                     | Repeat the encoding n times (to boost its signal)
        },                                |
        {                                 |
            "name": "types",              | Name of evidence ("types")
            "units": 32,                  | Size of the encoder hidden state
            "num_layers": 3               | Number of densely connected layers
            "tile": 1                     | Repeat the encoding n times (to boost its signal)
        },                                |
        {                                 |
            "name": "keywords",           | Name of evidence ("keywords")
            "units": 64,                  | Size of the encoder hidden state
            "num_layers": 3               | Number of densely connected layers
            "tile": 1                     | Repeat the encoding n times (to boost its signal)
        }                                 |
    ],                                    |
    "decoder": {                          | Provide parameters for the decoder here
        "units": 256,                     | Size of the decoder hidden state
        "num_layers": 3,                  | Number of layers in the decoder
        "max_ast_depth": 32               | Maximum depth of the AST (length of the longest path)
    }
    "reverse_encoder": {
        "units": 256,
        "num_layers": 3,
        "max_ast_depth": 32
    }                                   |
}                                         |
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
    reader = Reader(clargs, config)


    with tf.Session() as sess:
        predictor = model(clargs.save, sess, config) # goes to infer.BayesianPredictor
        reader.reset_batches()

        _psi_encoders = []
        _labels = []
        _rev_dict = {v: k for k, v in config.decoder.vocab.items()}

        for b in range(config.num_batches):
            # setup the feed dict
            ev_data, n, e, y = reader.next_batch()
            feed = {model.targets: y}
            for j, ev in enumerate(config.evidence):
                feed[model.encoder.inputs[j].name] = ev_data[j]
            for j in range(config.decoder.max_ast_depth):
                feed[model.decoder.nodes[j].name] = n[j]
                feed[model.decoder.edges[j].name] = e[j]
            # Feeding value into reverse encoder
                feed[model.reverse_encoder.nodes[j].name] = n[config.decoder.max_ast_depth - 1 - j]
                feed[model.reverse_encoder.edges[j].name] = e[config.decoder.max_ast_depth - 1 - j]

            _labels.extend(get_class_label(y,_rev_dict))
            # run the optimizer
            _psi_encoder \
                = sess.run(model.psi_encoder, feed)

            _psi_encoders.append(_psi_encoder)

        _psi_encoders_agg = np.concatenate(_psi_encoders, axis = 0)
        embedding(_psi_encoders_agg,_labels, sess)

def embedding(input_tensor, labels, sess):
   metadata = os.path.join(LOG_DIR, 'metadata.tsv')

   Embedding = tf.Variable( input_tensor , name='Embedding')


   with open(metadata, 'w') as metadata_file:
       for row in range(input_tensor.shape[0]):
           c=labels[row]
           metadata_file.write('{}\n'.format(c))

    saver = tf.train.Saver([Embedding])

    sess.run(images.initializer)
    saver.save(sess, os.path.join(LOG_DIR, 'Embedding.ckpt'))

    config = projector.ProjectorConfig()
    # One can add multiple embeddings.
    embedding = config.embeddings.add()
    embedding.tensor_name = images.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = metadata
    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)


def get_class_label(y, _rev_dict):
    _labels_batch = []
    for arr in y:
       flag = 0
       for val in arr:
          API_call = _rev_dict[val]
          for _class in _classes:
             if (API_call.find(_class)!= -1) == True:
                flag = 1
                _labels_batch.append(_class)
                break
          if flag == 1:
            break
       if flag == 0:
          _labels_batch.append('other_API')
    return _labels_batch

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
    clargs = parser.parse_args(['--save',
    '/home/ubuntu/bayou/src/main/python/bayou/models/low_level_evidences/save','/home/ubuntu/bayou/data/DATA-training.json'])


    sys.setrecursionlimit(clargs.python_recursion_limit)
    test(clargs)
