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

import time
import bayou.models.low_level_evidences.predict
from bayou.experiments.predictMethods.SearchDB.parallelReadJSON import parallelReadJSON
from bayou.experiments.predictMethods.SearchDB.searchFromDB import searchFromDB
from bayou.experiments.predictMethods.SearchDB.Embedding import EmbeddingBatch

class Predictor:

    def __init__(self, prob_mode=True):
        #set clargs.continue_from = True while testing, it continues from old saved config

        self.batch_size = 1000
        clargs.continue_from = True
        print('Loading Model, please wait _/\_ ...')
        model = bayou.models.low_level_evidences.predict.BayesianPredictor

        self.sess = tf.InteractiveSession()
        self.predictor = model(clargs.save, self.sess, batch_size=self.batch_size, prob_mode=prob_mode)# goes to predict.BayesianPredictor

        print ('Model Loaded, All Ready to Predict Evidences!!')
        sz = data_size = 100000
        print('Loading Data')
        self.nodes = np.zeros((sz, 8), dtype=np.int32)
        self.edges = np.zeros((sz, 8), dtype=np.bool)
        self.targets = np.zeros((sz, 8), dtype=np.int32) 
        self.ret_type =  np.zeros((sz, 1), dtype=np.int32) #np.zeros(self.inputs[4], num_batches, axis=0)
        self.formal_param =  np.zeros((sz, 8), dtype=np.int32) #np.split(self.inputs[5], num_batches, axis=0)
        num_batches = int(len(self.nodes)/self.predictor.config.batch_size)
        print('Done')

        # Batch
        self.ret_type = np.split(self.ret_type, num_batches, axis=0)
        self.formal_param = np.split(self.formal_param, num_batches, axis=0)
        self.nodes = np.split(self.nodes, num_batches, axis=0)
        self.edges = np.split(self.edges, num_batches, axis=0)
        self.targets = np.split(self.targets, num_batches, axis=0)
        return



class Decoder_Model:

    def __init__(self, predictor, mc_iter, topK=10):
        self.predictor = predictor
        self.mc_iter = mc_iter
        self.topK = topK
        return

   

    def get_running_comparison(self, psi):

        monteCarloIterations = self.mc_iter
        probY_iter = [None for i in range(monteCarloIterations)]
        sum_probY = [None for i in range(len(self.predictor.nodes))]
        total_decoder_time = 0.
        
        for mc_iter in range(monteCarloIterations):
            program_db = []
            start = time.perf_counter()
            for batch_num, (nodes, edges, targets, ret, fp) in enumerate(zip(self.predictor.nodes, self.predictor.edges, self.predictor.targets, self.predictor.ret_type, self.predictor.formal_param)):
                probYgivenZ = self.predictor.predictor.get_probY_given_psi(nodes, edges, targets, ret, fp, psi)
                if mc_iter == 0:
                    sum_probY[batch_num] = probYgivenZ
                else:
                    sum_probY[batch_num] = np.logaddexp(sum_probY[batch_num], probYgivenZ)
                batch_prob = sum_probY[batch_num] - np.log(mc_iter+1)


            end = time.perf_counter()
            mc_iter_time = (end-start)
            total_decoder_time += mc_iter_time

            top_progs = sorted(program_db, key=lambda x: x[3], reverse=True)
            print("MC Iteration :: " + str(mc_iter) + " Time :: " + str(total_decoder_time))



        return top_progs


class Rev_Encoder_Model:
    def __init__(self, numThreads=8, topK=10):
        self.numThreads = numThreads
        self.batch_size = 1
        self.minJSONs = 1
        self.maxJSONs = 2
        self.dimension = 256
        self.topK = topK
        self.scanner = self.get_database_scanner()
        return

    def get_database_scanner(self):

        JSONReader = parallelReadJSON('/home/ubuntu/DATABASE/', numThreads=self.numThreads, dimension=self.dimension, batch_size=self.batch_size, minJSONs=self.minJSONs , maxJSONs=self.maxJSONs)
        listOfColDB = JSONReader.readAllJSONs()
        scanner = searchFromDB(listOfColDB, self.topK, self.batch_size)
        return scanner


    def get_result(self, encA, encB, numThreads=None):

        if numThreads == None:
           numThreads = self.numThreads
        assert( self.numThreads % numThreads == 0)
        embIt_json = [{'a1':encA, 'b1':encB}]
     
        start = time.perf_counter()

        embIt_batch = EmbeddingBatch(embIt_json, 1, self.dimension)
        topKProgsBatch = self.scanner.searchAndTopKParallel(embIt_batch, numThreads = numThreads)
        topKProgs = topKProgsBatch[0]
        end = time.perf_counter()

        print("Rev Encoder Time :: " + str(end-start))

        #top_progs = sorted(topKProgs) #program_db, key=lambda x: x[3], reverse=True)#[:self.topK]
        return [] #top_progs

if __name__ == "__main__":
    #%%
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--python_recursion_limit', type=int, default=10000,
    help='set recursion limit for the Python interpreter')
    parser.add_argument('--save', type=str, default='/home/ubuntu/savedSearchModel')
    parser.add_argument('--mc_iter', type=int, default=10)

    clargs = parser.parse_args()
    sys.setrecursionlimit(clargs.python_recursion_limit)

    
    # initiate the server
    max_cut_off_accept = 10000
    pred = Predictor(prob_mode=False)
    
    psi = np.zeros((pred.batch_size, 256)) #encoder.get_latent_space(program)

    ## Please note that here cpu time is being used but programs are taken from TF. One reason is it is hard to get RT/FP from Program_output.json files if not indexed differently
    ## Second is that there are discrepancies in the result. This is most likely due to precision in a2 and b2 numbers stored in DB.
     
    rev_encoder_cpu = Rev_Encoder_Model(numThreads=1, topK=max_cut_off_accept)
    rev_encoder_top_progs_cpu = rev_encoder_cpu.get_result(-0.5, np.zeros((1,256)))
    
    decoder = Decoder_Model(pred, clargs.mc_iter, topK=max_cut_off_accept) #, golden_programs=rev_encoder_top_progs_cpu)
    decoder_top_progs = decoder.get_running_comparison(psi)
    
    print("=====================================")

