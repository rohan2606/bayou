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

class Get_Example_JSONs:
    def getExampleJsons(self, logdir, items):

        with open(logdir + '/L4TestProgramList.json', 'r') as f:
            js = json.load(f)
        programs = js['programs'][0:items]
        return programs


class Predictor:

    def __init__(self):
        #set clargs.continue_from = True while testing, it continues from old saved config
        clargs.continue_from = True
        print('Loading Model, please wait _/\_ ...')
        model = bayou.models.low_level_evidences.predict.BayesianPredictor

        sess = tf.InteractiveSession()
        self.predictor = model(clargs.save, sess) # goes to predict.BayesianPredictor

        print ('Model Loaded, All Ready to Predict Evidences!!')
        return

class Encoder_Model:

    def __init__(self, predictor):
        self.predictor = predictor
        return

    def get_latent_space(self, program):
        psi, EncA, EncB = self.predictor.get_latent_space(program)
        return psi, EncA, EncB



class Rev_Encoder_Model:
    def __init__(self):
        self.scanner = self.get_database_scanner()
        return

    def get_database_scanner(self):
        numThreads = 30
        batch_size = 1
        minJSONs = 1
        maxJSONs = 1
        dimension = 256
        topK = 11
        JSONReader = parallelReadJSON('/home/ubuntu/DATABASE/', numThreads=numThreads, dimension=dimension, batch_size=batch_size, minJSONs=minJSONs , maxJSONs=maxJSONs)
        listOfColDB = JSONReader.readAllJSONs()
        scanner = searchFromDB(listOfColDB, topK, batch_size)
        return scanner


    def get_result(self, encA, encB):
        embIt_batch = {['a1':encA, 'b1':encB]}

        topKProgsBatch = self.scanner.searchAndTopKParallel(embIt_batch, numThreads = numThreads)
        topKProgs = topKProgsBatch[0]
        return [prog.body for prog in topKProgs]



class Decoder_Model:

    def __init__(self, predictor):
        self.predictor = predictor
        num_batches = 50

        # Load
        self.nodes = np.load('../../../models/low_level_evidences/data/nodes.npy')
        self.edges = np.load('../../../models/low_level_evidences/data/edges.npy')
        self.targets = np.load('../../../models/low_level_evidences/data/targets.npy')

        # Batch
        self.nodes = np.split(self.nodes, num_batches, axis=0)
        self.edges = np.split(self.edges, num_batches, axis=0)
        self.targets = np.split(self.targets, num_batches, axis=0)
        return


    def get_ProbYs_given_X(self, program, monteCarloIterations = 10):

        sum_probY = None
        for batch_num, nodes, edges, targets in enumerate(zip(self.nodes, self.edges, self.targets)):
            for count in range(monteCarloIterations):
                probYgivenZ = self.predictor.get_probY_given_psi(nodes, edges, targets, psi)
                if count == 0:
                    sum_probY = probYgivenZ
                else:
                    sum_probY = np.logaddexp(sum_probY, probYgivenZ)
            batch_prob = sum_probY - np.log(monteCarloIterations)
        return







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
    pred = Predictor()
    encoder = Encoder_Model(pred)
    decoder = Decoder_Model(pred)
    rev_encoder = Rev_Encoder_Model()

    # get the input JSON
    programs = Get_Example_JSONs.getExampleJsons('../predictMethods/log/expNumber_6/', 1)
    #get the probs
    j=0
    program = programs[j]
    print("Working with program no :: " + str(j))
    psi = encoder.get_latent_space(program)
    rev_encoder_top_progs = rev_encoder.get_result(probYs)
    print(rev_encoder_top_progs)
    probYs = decoder.get_ProbYs_given_X(program)




    # get probs from trained model
