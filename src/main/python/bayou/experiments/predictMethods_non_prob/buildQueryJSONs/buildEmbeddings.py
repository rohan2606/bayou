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


import time
import bayou.models.non_prob.predict

#%%
parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--python_recursion_limit', type=int, default=10000,
                    help='set recursion limit for the Python interpreter')
parser.add_argument('--save', type=str, default='/home/ubuntu/savedSearchModel_non_prob',
                    help='checkpoint model during training here')

clargs = parser.parse_args()
sys.setrecursionlimit(clargs.python_recursion_limit)



class embedding_server():
    def __init__(self):
        #set clargs.continue_from = True while testing, it continues from old saved config
        clargs.continue_from = True
        print('Loading Model, please wait _/\_ ...')
        model = bayou.models.non_prob.predict.BayesianPredictor

        sess = tf.InteractiveSession()
        self.predictor = model(clargs.save, sess) # goes to predict.BayesianPredictor

        print ('Model Loaded, All Ready to Predict Evidences!!')

        return


    def getEmbeddings(self, logdir):
        with open(logdir + '/L4TestProgramList.json', 'r') as f:
            js = json.load(f)

        programs = []
        count = 0
        for program in js['programs']:
            psi_enc = self.predictor.get_psi_encoder(program)
            program['ev_psi'] = [val.item() for val in psi_enc[0]]
            programs.append(program)

        print('\nWriting to {}...'.format(''), end='\n')
        with open(logdir + '/EmbeddedProgramList.json', 'w') as f:
            json.dump({'embeddings': programs}, fp=f, indent=2)

        return
