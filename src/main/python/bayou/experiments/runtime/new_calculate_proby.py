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
import pickle

import time
import ijson
from bayou.experiments.predictMethods.SearchDB.parallelReadJSON import parallelReadJSON
from bayou.experiments.predictMethods.SearchDB.searchFromDB import searchFromDB
from bayou.experiments.predictMethods.SearchDB.Embedding import EmbeddingBatch
from bayou.models.low_level_evidences.test import get_c_minus_cstar

from bayou.experiments.predictMethods.SearchDB.utils import get_jaccard_distace_api


import bayou.models.low_level_evidences.predict
from bayou.models.low_level_evidences.test import get_c_minus_cstar

class Get_Example_JSONs:

    def getExampleJsons(logdir, items):

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
        self.predictor = model(clargs.save, sess, batch_size=500)# goes to predict.BayesianPredictor

        print ('Model Loaded, All Ready to Predict Evidences!!')

        print('Loading Data')
        self.nodes = np.load('../../models/low_level_evidences/data/nodes.npy')
        self.edges = np.load('../../models/low_level_evidences/data/edges.npy')
        self.targets = np.load('../../models/low_level_evidences/data/targets.npy')
        with open('../../models/low_level_evidences/data/inputs.npy', 'rb') as f:
            self.inputs = pickle.load(f)

        num_batches = int(len(self.nodes)/self.predictor.config.batch_size)
        self.js_programs = []
        with open('../../models/low_level_evidences/data/js_programs.json', 'rb') as f:
            for program in ijson.items(f, 'programs.item'):
                self.js_programs.append(program)
        print('Done')
        # Batch
        self.ret_type = np.split(self.inputs[4], num_batches, axis=0)
        self.formal_param = np.split(self.inputs[5], num_batches, axis=0)
        self.nodes = np.split(self.nodes, num_batches, axis=0)
        self.edges = np.split(self.edges, num_batches, axis=0)
        self.targets = np.split(self.targets, num_batches, axis=0)
        self.js_programs = self.chunks(self.js_programs, self.predictor.config.batch_size)
        return

    def chunks(self, l, n):
        """Yield successive n-sized chunks from l."""
        chunks = []
        for i in range(0, len(l), n):
            chunks.append(l[i:i + n])
        return chunks

class Encoder_Model:

    def __init__(self, predictor):
        self.predictor = predictor
        return

    def get_latent_space(self, program):
        psi, EncA, EncB = self.predictor.predictor.get_psi_encoder(program)
        return psi, EncA, EncB


class Rev_Encoder_Model_2:
    def __init__(self):
        return


    def get_result(self, encA, encB):

        program_db = []
        sum_probY = None
        for batch_num, (nodes, edges, targets, ret, fp, jsons) in enumerate(zip(self.predictor.nodes, self.predictor.edges, self.predictor.targets, self.predictor.ret_type, self.predictor.formal_param, self.predictor.js_programs)):
            probY, RevEncA, RevEncB = self.predictor.predictor.get_rev_enc_ab(nodes, edges, targets, ret, fp)
            batch_prob = get_c_minus_cstar(encA, encB, RevEncA, RevEncB, probY, self.predictor.config.latent_size)

            for i, js in enumerate(jsons):
                 program_db.append((js['body'], batch_prob[i]))
            #if batch_num > 200:
            #   break
            print(f'Batch# {batch_num}/{len(self.nodes)}',end='\r')

        top_progs = sorted(program_db, key=lambda x: x[1], reverse=True)[:10]
        return top_progs


class Rev_Encoder_Model:
    def __init__(self):
        self.numThreads = 30
        self.batch_size = 1
        self.minJSONs = 2000
        self.maxJSONs = 2001
        self.dimension = 256
        self.topK = 100
        self.scanner = self.get_database_scanner()
        return

    def get_database_scanner(self):

        JSONReader = parallelReadJSON('/home/ubuntu/DATABASE/', numThreads=self.numThreads, dimension=self.dimension, batch_size=self.batch_size, minJSONs=self.minJSONs , maxJSONs=self.maxJSONs)
        listOfColDB = JSONReader.readAllJSONs()
        scanner = searchFromDB(listOfColDB, self.topK, self.batch_size)
        return scanner


    def get_result(self, encA, encB):
        embIt_json = [{'a1':encA, 'b1':encB}]
        embIt_batch = EmbeddingBatch(embIt_json, 1, 256)
        topKProgsBatch = self.scanner.searchAndTopKParallel(embIt_batch, numThreads = self.numThreads)
        topKProgs = topKProgsBatch[0]
        return [prog.body for prog in topKProgs]


class Decoder_Model:

    def __init__(self, predictor, mc_iter):
        self.predictor = predictor
        self.mc_iter = mc_iter

        # Load


    def get_running_comparison(self, program, psis, golden_programs=list()):

        monteCarloIterations = self.mc_iter
        sum_probY = [None for i in range(len(self.predictor.nodes))]
        for mc_iter in range(monteCarloIterations):
            psi = np.tile(psis[mc_iter],(self.predictor.predictor.config.batch_size,1))
            program_db = []
            for batch_num, (nodes, edges, targets, ret, fp, jsons) in enumerate(zip(self.predictor.nodes, self.predictor.edges, self.predictor.targets, self.predictor.ret_type, self.predictor.formal_param, self.predictor.js_programs)):
                probYgivenZ = self.predictor.predictor.get_probY_given_psi(nodes, edges, targets, ret, fp, psi)
                if mc_iter == 0:
                    sum_probY[batch_num] = probYgivenZ
                else:
                    sum_probY[batch_num] = np.logaddexp(sum_probY[batch_num], probYgivenZ)
                batch_prob = sum_probY[batch_num] - np.log(mc_iter+1)

                for i, js in enumerate(jsons):
                     program_db.append((js['body'], batch_prob[i]))

            top_progs = sorted(program_db, key=lambda x: x[1], reverse=True)[:100]

            count = 0
            for top_prog in top_progs:
                if top_prog[0] in golden_programs:
                    count += 1
            distance100 = count / len(golden_programs)
            print(f"Monte Carlo Iteration: {mc_iter}, Distance[1/3/5/10/100]: {distance100}/")
            #print(f"Monte Carlo Iteration: {mc_iter}, Distance[1/3/5/10/100]: {distance1}/{distance3}/{distance5}/{distance10}/{distance100}/")
            print("=====================================")
        return top_progs




if __name__ == "__main__":
    #%%
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--python_recursion_limit', type=int, default=10000,
    help='set recursion limit for the Python interpreter')
    parser.add_argument('--save', type=str, default='/home/ubuntu/savedSearchModel')
    parser.add_argument('--mc_iter', type=int, default=1)

    clargs = parser.parse_args()
    sys.setrecursionlimit(clargs.python_recursion_limit)

    # initiate the server
    pred = Predictor()
    encoder = Encoder_Model(pred)
    decoder = Decoder_Model(pred, clargs.mc_iter)
    #rev_encoder = Rev_Encoder_Model_2(pred)
    rev_encoder = Rev_Encoder_Model()

    # get the input JSON
    programs = Get_Example_JSONs.getExampleJsons('../predictMethods/log/expNumber_6/', 10)
    max_cut_off_accept = 100
    #get the probs
    j=0
    program = programs[j]
    print(program)
    print("Working with program no :: " + str(j))
    psis = []
    while(len(psis) < clargs.mc_iter):
         psi, eA, eB = encoder.get_latent_space(program)
         psi = np.vsplit(psi, len(psi))
         psis.extend(psi)
    rev_encoder_top_progs = rev_encoder.get_result(eA[0], eB[0])[:max_cut_off_accept]

    #for top_prog in rev_encoder_top_progs:
    #    print(top_prog)
    print("=====================================")

    decoder_top_progs = decoder.get_running_comparison(program, psis, golden_programs=rev_encoder_top_progs)
    # for top_prog in decoder_top_progs:
    #     print(top_prog[0])
    #     print(top_prog[1])
    #
    # print("=====================================")

