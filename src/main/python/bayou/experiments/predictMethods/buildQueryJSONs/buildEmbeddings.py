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
import bayou.models.low_level_evidences.predict

#%%
parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--python_recursion_limit', type=int, default=10000,
                    help='set recursion limit for the Python interpreter')
parser.add_argument('--save', type=str, default='/home/ubuntu/savedSearchModel',
                    help='checkpoint model during training here')

clargs = parser.parse_args()
sys.setrecursionlimit(clargs.python_recursion_limit)



class embedding_server():
    def __init__(self):
        #set clargs.continue_from = True while testing, it continues from old saved config
        clargs.continue_from = True
        print('Loading Model, please wait _/\_ ...')
        model = bayou.models.low_level_evidences.predict.BayesianPredictor

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
            a1, b1, a2, b2, probY,ignored = self.predictor.get_a1b1a2b2(program)


            if ignored == True:
                continue

            program['a1'] = a1[0].item() # .item() converts a numpy element to a python element, one that is JSON serializable
            program['b1'] = [val.item() for val in b1[0]]
            program['a2'] = a2[0].item()
            program['b2'] = [val.item() for val in b2[0]]
            program['ProbY'] = probY[0].item()
            programs.append(program)

        print('\nWriting to {}...'.format(''), end='\n')
        with open(logdir + '/EmbeddedProgramList.json', 'w') as f:
            json.dump({'embeddings': programs}, fp=f, indent=2)
    
        return
