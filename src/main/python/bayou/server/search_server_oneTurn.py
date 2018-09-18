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
#


from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import os
import sys
import json
import textwrap
import socket

import time
import bayou.models.low_level_evidences.predict

File_Name = 'Search_Data_Basic'

HELP = """ Help me! :( """
#%%


def search_server(clargs):
    #set clargs.continue_from = True while testing, it continues from old saved config
    clargs.continue_from = True
    print('Loading Model, please wait _/\_ ...')
    model = bayou.models.low_level_evidences.predict.BayesianPredictor


    with tf.Session() as sess:
        predictor = model(clargs.save, sess) # goes to predict.BayesianPredictor
        print ('Model Loaded, All Ready to Predict Evidences!!')

        with open(clargs.queryProg, 'r') as f:
            js = json.load(f)
        a1, b1 = predictor.get_a1b1(js['programs'][0])
        # evSigmas = predictor.get_ev_sigma(js['programs'][0])

        # print(evSigmas)
        # program = jsp[0]
        # We do not need other paths in the program as all the evidences are the same for all the paths
        # and for new test code we are only interested in the evidence encodings
        # a1, a2 and ProbY are all scalars, b1 and b2 are vectors

        programs = []
        program = {}
        program['a1'] = a1[0].item() # .item() converts a numpy element to a python element, one that is JSON serializable
        program['b1'] = [val.item() for val in b1[0]]


        programs.append(program)

        for j in range(int(np.ceil(num_progs / batch_size))):
            sid, eid = j * batch_size, min( (j+1) * batch_size , num_progs)
            prob_Y_X = get_c_minus_cstar(np.array(program['a1']), np.array(program['b1']),\
                                    np.array(a2s[sid:eid]), np.array(b2s[sid:eid]), np.array(prob_Ys[sid:eid]), latent_size)
            prob_Y_Xs += list(prob_Y_X)

        _rank = find_my_rank( prob_Y_Xs , i )


    return



#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent(HELP))
    # parser.add_argument('input_file', type=str, nargs=1,
    #                     help='input data file')
    parser.add_argument('--python_recursion_limit', type=int, default=10000,
                        help='set recursion limit for the Python interpreter')
    parser.add_argument('--save', type=str, default='savedSearchModel',
                        help='checkpoint model during training here')
    parser.add_argument('--evidence', type=str, default='all',
                        choices=['apicalls', 'types', 'keywords', 'all'],
                        help='use only this evidence for inference queries')
    parser.add_argument('--queryProg', type=str, required=True,
                        help='query prog to be parsed')
    parser.add_argument('--database', type=str, required=True,
                        help='database of codes')
    parser.add_argument('--output_file', type=str, default=None,
                        help='output file to print probabilities')

    #clargs = parser.parse_args()
    clargs = parser.parse_args()

    sys.setrecursionlimit(clargs.python_recursion_limit)
    search_server(clargs)
