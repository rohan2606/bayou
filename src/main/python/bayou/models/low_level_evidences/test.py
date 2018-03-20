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

from bayou.models.low_level_evidences.data_reader import Reader
from bayou.models.low_level_evidences.model import Model
from bayou.models.low_level_evidences.utils import read_config, dump_config

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
    if model_type == 'core':
        model = bayou.models.core.infer.BayesianPredictor
    elif model_type == 'lle':
        model = bayou.models.low_level_evidences.infer.BayesianPredictor
    else:
        raise ValueError('Invalid model type in config: ' + model_type)

    config.batch_size = 1 # this is to make sure that every code search worls in series, might be easier to do in batches later
    reader = Reader(clargs, config)

    with tf.Session() as sess:
        predictor = model(clargs.save,sess) # goes to infer.BayesianPredictor

        # testing
        reader.reset_batches()

        # computing P(Y) = int_Z P(Z,Y) = int_Z(P(Y|Z)*P(Z)) = int_Z(P(Y|Z)*P(Z|X)*P(Z)/P(Z|X))
        # = 1/L Sum_i={1,L} P(Y|Z_i)*P(Z_i)/P(Z_i|X) where Z_i ~ P(Z|X) = N(0,1)
        prob_Y, a1b1, a2b2 = [], [], []
        start = time.time()
        for i in range(config.num_batches):
            # setup the feed dict, ignoring the evidences and raw_targets
            ev_data, n, e, y = reader.next_batch()
            prob_Y.append(predictor.get_Prob_Y_i(ev_data, n, e, y))
            a1b1.append(predictor.get_encoder_abc(ev_data))
            a2b2.append(predictor.get_rev_encoder_abc(n,e))


        end = time.time()

        reader.reset_batches()
        avg_loss = 0
        avg_gen_loss = 0


        for i in range(config.num_batches):
            prob_Y_X = []
            for j in range(config.num_batches)
                prob_Y_X_i = predictor.get_PY_given_Xi(a1b1[i], a2b2[j]) * prob_Y[j]
                prob_Y_X.append(prob_Y_X_i)
            sort_id = np.argsort(prob_Y_X).index(i)
            if sort_id < 5:
                print('Success')
            else:
                print('Fail')




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
    clargs = parser.parse_args(['--config','config.json',
    '..\..\..\..\..\..\data\DATA-training-top.json'])

    sys.setrecursionlimit(clargs.python_recursion_limit)
    test(clargs)
