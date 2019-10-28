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
import re

import time
import ijson
from bayou.experiments.predictMethods_non_prob.SearchDB.parallelReadJSON import parallelReadJSON
from bayou.experiments.predictMethods_non_prob.SearchDB.searchFromDB import searchFromDB
from bayou.experiments.predictMethods_non_prob.SearchDB.Embedding import EmbeddingBatch

from bayou.experiments.predictMethods_non_prob.SearchDB.utils import get_jaccard_distace_api


import bayou.models.non_prob.predict
from bayou.experiments.human_input.extract_evidence import extract_evidence

import subprocess

class Java_Reader:


    def useDomDriver(filepath):
        subprocess.call(['java', '-jar', \
        '/home/ubuntu/bayou/tool_files/maven_3_3_9/dom_driver/target/dom_driver-1.0-jar-with-dependencies.jar', \
        '-f', filepath, '-c', '/home/ubuntu/bayou/Java-prog-extract-config.json', \
        '-o', 'problems/output.json'])
        return

    def getExampleJsons(logdir, items):

        extracted_ev = extract_evidence(logdir)

        return extracted_ev



class Predictor:

    def __init__(self):
        #set clargs.continue_from = True while testing, it continues from old saved config
        clargs.continue_from = True
        print('Loading Model, please wait _/\_ ...')
        model = bayou.models.non_prob.predict.BayesianPredictor

        sess = tf.InteractiveSession()
        self.predictor = model(clargs.save, sess) #, batch_size=500)# goes to predict.BayesianPredictor

        print ('Model Loaded, All Ready to Predict Evidences!!')

        return


class Encoder_Model:

    def __init__(self, predictor):
        self.predictor = predictor
        return

    def get_latent_space(self, program):
        psi_enc = self.predictor.predictor.get_psi_encoder(program)
        return psi_enc


class Rev_Encoder_Model:
    def __init__(self, db_location):
        self.numThreads = 30
        self.batch_size = 1
        self.minJSONs = 1
        self.maxJSONs = 10
        self.dimension = 256
        self.topK = 10
        self.db_location = db_location
        self.scanner = self.get_database_scanner()
        return

    def get_database_scanner(self):

        JSONReader = parallelReadJSON(self.db_location, numThreads=self.numThreads, dimension=self.dimension, batch_size=self.batch_size, minJSONs=self.minJSONs , maxJSONs=self.maxJSONs)
        listOfColDB = JSONReader.readAllJSONs()
        scanner = searchFromDB(listOfColDB, self.topK, self.batch_size)
        return scanner


    def get_result(self, ev_psi):
        embIt_json = [{'ev_psi':ev_psi}]
        embIt_batch = EmbeddingBatch(embIt_json, 1, 256)
        topKProgsBatch = self.scanner.searchAndTopKParallel(embIt_batch, numThreads = self.numThreads)
        topKProgs = topKProgsBatch[0]
        return [prog.body for prog in topKProgs]





if __name__ == "__main__":
    #%%
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--python_recursion_limit', type=int, default=10000,
    help='set recursion limit for the Python interpreter')
    parser.add_argument('input_file', type=str, nargs=1,
                        help='input data file')
    parser.add_argument('--save', type=str, default='/home/ubuntu/save_500_new_drop_skinny_seq')
    parser.add_argument('--db_location', type=str, default='/home/ubuntu/DATABASE/')
    parser.add_argument('--mc_iter', type=int, default=1)

    clargs = parser.parse_args()
    sys.setrecursionlimit(clargs.python_recursion_limit)

    print(clargs.input_file)
    # initiate the server
    pred = Predictor()
    encoder = Encoder_Model(pred)
    #rev_encoder = Rev_Encoder_Model_2(pred)
    rev_encoder = Rev_Encoder_Model(clargs.db_location)

    # get the input JSON
    filename = clargs.input_file[0]
    Java_Reader.useDomDriver(filename)

    programs = Java_Reader.getExampleJsons('problems/output.json',10)

    print(programs)
    print("=====================================")
    #get the probs

    print("=====================================")
    ev_psi = encoder.get_latent_space(json.loads(programs))
    rev_encoder_top_progs = rev_encoder.get_result(ev_psi[0])[:rev_encoder.topK]

    for j, top_prog in enumerate(rev_encoder_top_progs):
       print('Rank ::' +  str(j))
       print(top_prog)
    print("=====================================")
