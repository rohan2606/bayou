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

TCP_IP = '127.0.0.1'
TCP_PORT = 5005
BUFFER_SIZE = 1024 #Normally 1024, but we want fast response

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))


def search_server(clargs):
    #set clargs.continue_from = True while testing, it continues from old saved config
    clargs.continue_from = True
    print('Loading Model, please wait _/\_ ...')
    model = bayou.models.low_level_evidences.predict.BayesianPredictor


    with tf.Session() as sess:
        predictor = model(clargs.save, sess) # goes to predict.BayesianPredictor

        print ('Model Loaded, All Ready to Predict Evidences!!')
        while True:
            print("\n\n Waiting for a new connection!")
            s.listen(1)
            conn, addr = s.accept()
            print ('Connection address:', addr)
            while True:
                data = conn.recv(BUFFER_SIZE)
                if not data:  break

                with open('/home/ubuntu/QueryProg.json', 'r') as f:
                    js = json.load(f)
                a1, b1 = predictor.get_a1b1(js['programs'][0])
                evSigmas = predictor.get_ev_sigma(js['programs'][0])

                print(evSigmas)
                # program = jsp[0]
                # We do not need other paths in the program as all the evidences are the same for all the paths
                # and for new test code we are only interested in the evidence encodings
                # a1, a2 and ProbY are all scalars, b1 and b2 are vectors

                programs = []
                program = {}
                program['a1'] = a1[0].item() # .item() converts a numpy element to a python element, one that is JSON serializable
                program['b1'] = [val.item() for val in b1[0]]


                programs.append(program)

                print('\nWriting to {}...'.format('/home/ubuntu/QueryProgWEncoding.json'), end='\n')
                with open('/home/ubuntu/QueryProgWEncoding.json', 'w') as f:
                    json.dump({'programs': programs}, fp=f, indent=2)
                print ("Received data from client:", data)
                conn.send(data)  # echo

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
    parser.add_argument('--output_file', type=str, default=None,
                        help='output file to print probabilities')

    #clargs = parser.parse_args()
    clargs = parser.parse_args()
	# [
    #  # '..\..\..\..\..\..\data\DATA-training-top.json'])
    #  #'/home/rm38/Research/Bayou_Code_Search/Corpus/DATA-training-expanded-biased-TOP.json'])
    #  # '/home/ubuntu/Corpus/DATA-training-expanded-biased.json'])
    #  '/home/ubuntu/QueryProg.json'])
    sys.setrecursionlimit(clargs.python_recursion_limit)
    search_server(clargs)
