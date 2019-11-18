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
from bayou.experiments.predictMethods.SearchDB.parallelReadJSON import parallelReadJSON
from bayou.experiments.predictMethods.SearchDB.searchFromDB import searchFromDB
from bayou.experiments.predictMethods.SearchDB.Embedding import EmbeddingBatch


import socket

from extract_evidence import extract_evidence

import subprocess

class Java_Reader:


    def useDomDriver(filepath):
        subprocess.call(['java', '-jar', \
        '/home/ubuntu/bayou/tool_files/maven_3_3_9/dom_driver/target/dom_driver-1.0-jar-with-dependencies.jar', \
        '-f', filepath, '-c', '/home/ubuntu/bayou/Java-prog-extract-config.json', \
        '-o', 'log/output.json'])
        return

    def getExampleJsons(logdir, items):

        extracted_ev = extract_evidence(logdir)

        return extracted_ev





class Rev_Encoder_Model:
    def __init__(self, batch_size=1, topK=10):
        self.numThreads = 30
        self.batch_size = batch_size
        self.minJSONs = 1
        self.maxJSONs =  10
        self.dimension = 256
        self.topK = topK
        self.scanner = self.get_database_scanner()
        return

    def get_database_scanner(self):

        JSONReader = parallelReadJSON('/home/ubuntu/DATABASE/', numThreads=self.numThreads, dimension=self.dimension, batch_size=self.batch_size, minJSONs=self.minJSONs , maxJSONs=self.maxJSONs)
        listOfColDB = JSONReader.readAllJSONs()
        scanner = searchFromDB(listOfColDB, self.topK, self.batch_size)
        return scanner


    def get_result(self, encAs, encBs):
        
        embIt_json = []
        for encA, encB in zip(encAs, encBs):
             embIt_json.append({'a1':encA, 'b1':encB})

        embIt_batch = EmbeddingBatch(embIt_json, self.batch_size, 256)
        topKProgsBatch = self.scanner.searchAndTopKParallel(embIt_batch, numThreads = self.numThreads)
        return [[(prog[0].body, prog[1]) for prog in topKProgs[:self.topK]] for topKProgs in topKProgsBatch]


def client_socket(st):
	client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	client.connect((socket.gethostname(), 5000))
	byt = st.encode()
	client.send(byt)
	from_server = client.recv(1000000)
	from_server.decode()
	client.close()
	return from_server


if __name__ == "__main__":
    #%%
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--python_recursion_limit', type=int, default=10000,
    help='set recursion limit for the Python interpreter')
    parser.add_argument('input_file', type=str, nargs=1,
                        help='input data file')
    parser.add_argument('--topK', type=int, default=10)

    clargs = parser.parse_args()
    sys.setrecursionlimit(clargs.python_recursion_limit)



    if not os.path.exists('log'):
       os.makedirs('log')

    eAs, eBs = [], []
    programs = []
    if os.path.isfile(clargs.input_file[0]):
        qry_files = [clargs.input_file[0]]
    elif os.path.isdir(clargs.input_file[0]):
        qry_files = []
        for filename in sorted(os.listdir(clargs.input_file[0])):
            if os.path.isfile(clargs.input_file[0] + '/' + filename):
                 qry_files.append(clargs.input_file[0] + '/' + filename)
    
    for j, filename in enumerate(qry_files): 
    	Java_Reader.useDomDriver(filename)

    	input_qry_lines = open(filename).read().split('\n')
    	output_qry_lines = []
    	for line in input_qry_lines:
    	    out_line = line.replace('PDB_FILL','CODE_SEARCH')
    	    out_line = out_line.replace('None','<?>')
    	    output_qry_lines.append(out_line + '\n')

    	log_folder = 'log/' + 'qry_' + str(j) + '/'
    	if not os.path.exists(log_folder):
    	    os.makedirs(log_folder)
    	with open(log_folder + 'query.java', 'w') as f:
    	     f.writelines(output_qry_lines)


    	program = Java_Reader.getExampleJsons( 'log/output.json',10)


    	with open(log_folder + 'output_wSurr.json', 'w') as f:
    	     json.dump(json.loads(program), f, indent=4)
        
    	server_data = client_socket(program)
    	server_data = json.loads(server_data)
    	eA, eB = server_data['eA'], server_data['eB']
    	eAs.append(eA)
    	eBs.append(eB)
    	programs.append(program)

    #rev_encoder = Rev_Encoder_Model_2(pred)
    rev_encoder = Rev_Encoder_Model(batch_size = len(eAs), topK=clargs.topK)
    rev_encoder_batch_top_progs = rev_encoder.get_result(eAs, eBs)

    for j, rev_encoder_top_progs in enumerate(rev_encoder_batch_top_progs):
    	print(programs[j])
    	for i, top_prog in enumerate(rev_encoder_top_progs):
    	   print('Rank ::' +  str(i))
    	   print('Prob ::' + str(top_prog[1]))
    	   print(top_prog[0])
    	   with open('log/qry_' + str(j) + '/program'+str(i)+'.java','w') as f:
    	        f.write(top_prog[0])
    	print("=====================================")

    os.remove('log/output.json')
