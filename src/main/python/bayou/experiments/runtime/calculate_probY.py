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

    def manto_carlo_prob_y_given_x(self, qry_program, db_program_items, monteCarloIterations = 10):

        def single_prob(nodes, edges, targets, inputs):
            sum_probY = None
            for count in range(monteCarloIterations):
                probYgivenX = self.predictor.get_probYgivenX(nodes, edges, targets, inputs)
                if count == 0:
                    sum_probY = probYgivenX
                else:
                    sum_probY = np.logaddexp(sum_probY, probYgivenX)

            return sum_probY - np.log(monteCarloIterations)

        _, _, _, inputs = self.predictor.wrangle_data(qry_program)
        prob_Y_given_X = [None]*len(db_program_items[0])

        k=0
        for node, edge, target in zip(*db_program_items):
            prob_Y_given_X[k] = single_prob(node, edge, target, inputs)
            k+=1
        Z = np.logaddexp.reduce(np.array(prob_Y_given_X),dtype=float)
        return ["%.2f" % (prob.item() - Z) for prob in prob_Y_given_X]



    def rev_encoder_prob_y_given_x(self, qry_program, db_programs):
        EncA, EncB, _, _, _, ignored= self.predictor.get_a1b1a2b2(qry_program)
        prob_Y_given_X = [None]*len(db_programs)
        prob_Y = [None]*len(db_programs)
        for k, program in enumerate(db_programs):
            _ , _ , RevEncA, RevEncB, probY, ignored= self.predictor.get_a1b1a2b2(program)
            prob_Y_given_X[k], prob_Y[k] = get_c_minus_cstar(EncA[0], EncB[0], RevEncA[0:1], RevEncB[0:1], probY[0:1], self.predictor.config.latent_size)
        Z1 = np.logaddexp.reduce(np.array(prob_Y_given_X),dtype=float)
        Z2 = np.logaddexp.reduce(np.array(prob_Y),dtype=float)
        #return ["%.2f" % (prob1.item() - Z1 + prob2 - Z2) for prob1, prob2 in zip(prob_Y_given_X, prob_Y)]
        return ["%.2f" % (prob1.item() - Z1) for prob1 in prob_Y_given_X], ["%.2f" % (prob2.item()) for prob2 in prob_Y]





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
    all_programs = EmbS.getExampleJsons('../predictMethods/log/expNumber_6/', 10)
    db_program = all_programs[:10]

    nodes, edges, targets, inputs = [], [], [], []
    for program in db_program:
        node, edge, target, input_ = EmbS.predictor.wrangle_data(program)
        nodes.append(node)
        edges.append(edge)
        targets.append(target)
        inputs.append(input_)

    db_program_items = [nodes, edges, targets]
    qry_program_inputs = inputs
    # print(len(db_program))
    #get the probs
    # std_errors = []
    mc_iter = 1
    prob_Y = [[] for i in range(len(db_program))]
    for k, program in enumerate(db_program):
        for iter in range(mc_iter):
            _ , _ , _, _, temp_prob_Y, _ = EmbS.predictor.get_a1b1a2b2(program)
            prob_Y[k].append(temp_prob_Y)
    for i in range(len(prob_Y)):
        prob_Y[i] = np.logaddexp.reduce(np.array(prob_Y[i]),dtype=float) - np.log(mc_iter)
        
    
    print(["%.2f"%p.item() for p in prob_Y])
    sys.exit(0)
    for j, qry_program in enumerate(all_programs):
        print("Working with program no :: " + str(j))
        rev_prob, prob_y = EmbS.rev_encoder_prob_y_given_x(qry_program, db_program)
        mc_mean = EmbS.manto_carlo_prob_y_given_x(qry_program, db_program_items)

        print([(a,b) for a,b in zip(rev_prob, prob_y)])
          
        print(mc_mean)

    # final_error_graph = np.mean(np.stack(std_errors, axis=0), axis=0)

    # get probs from trained model

    # print(final_error_graph)
        # get_estimate_at_step
