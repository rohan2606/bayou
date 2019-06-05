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
from bayou.models.low_level_evidences.test import get_c_minus_cstar



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

    def getExampleJsons(self, logdir, items):

        with open(logdir + '/L4TestProgramList.json', 'r') as f:
            js = json.load(f)

        probYs = []
        programs = js['programs'][0:items]
        return programs

    def get_ProbYs_given_X(self, program, monteCarloIterations = 1000):

        nodes, edges, inputs = self.predictor.wrangle_data(program)
        probYs = []
        for count in range(monteCarloIterations):
            # print(count)
            probYgivenX = self.predictor.get_probYgivenX(nodes, edges, inputs)
            probYs.append(probYgivenX)

        # idealProbY = np.mean(np.asarray(probYs))
        return probYs


    def get_ProbYs_given_X_from_Bayou(self, program):

        EncA, EncB, RevEncA, RevEncB, probY, ignored= self.predictor.get_a1b1a2b2(program)
        # idealProbY = np.mean(np.asarray(probYs))
        prob_Y_given_X = get_c_minus_cstar(EncA, EncB, RevEncA, RevEncB, probY, self.predictor.config.latent_size)
        return prob_Y_given_X

    def get_running_error(self, probYs):

        running_mean = []
        curr_sum = 0
        for count , probY in enumerate(probYs):
            curr_sum += probY
            curr_mean = curr_sum/(count+1)
            running_mean.append(curr_mean)

        ideal_val = running_mean[-1]

        running_std_errors = []
        for val in running_mean:
            std_error = np.sqrt( np.square(ideal_val - val) )
            running_std_errors.append(std_error)

        return np.asarray(running_std_errors)




if __name__ == "__main__":
    #%%
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--python_recursion_limit', type=int, default=10000,
    help='set recursion limit for the Python interpreter')
    parser.add_argument('--save', type=str, default='/home/ubuntu/savedSearchModel',
    help='checkpoint model during training here')

    clargs = parser.parse_args()
    sys.setrecursionlimit(clargs.python_recursion_limit)

    # initiate the server
    EmbS = embedding_server()
    # get the input JSON
    programs = EmbS.getExampleJsons('../predictMethods/log/expNumber_6/', 10)
    #get the probs
    std_errors = []
    for j, program in enumerate(programs):
        print("Working with program no :: " + str(j))
        probYs = EmbS.get_ProbYs_given_X(program)
        running_std_errors = EmbS.get_running_error(probYs)
        std_errors.append(running_std_errors)
        probY_model = EmbS.get_ProbYs_given_X_from_Bayou(program)
        print(probY_model)
        print(probYs[-1])

    final_error_graph = np.mean(np.stack(std_errors, axis=0), axis=0)

    # get probs from trained model

    print(final_error_graph)
        # get_estimate_at_step
