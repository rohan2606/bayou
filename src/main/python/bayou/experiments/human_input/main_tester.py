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

import argparse
import os
import sys
import json
import pickle
import re

import time
import ijson


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



def client_socket_database(st):
	client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	client.connect((socket.gethostname(), 5001))
	byt = st.encode()
	client.send(byt)
	from_server = client.recv(1000000)
	from_server.decode()
	client.close()
	return from_server




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
    

    all_qry_progs = {'eAs':eAs,'eBs':eBs}
    server_data = client_socket_database(json.dumps(all_qry_progs))
    server_data.decode()
    print(server_data)
    #rev_encoder = Rev_Encoder_Model_2(pred)

