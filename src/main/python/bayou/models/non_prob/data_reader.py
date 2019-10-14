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

from __future__ import print_function
import json
import ijson.backends.yajl2_cffi as ijson
import numpy as np
import random
import os
import _pickle as pickle
from collections import Counter
import gc
import copy

from bayou.models.low_level_evidences.utils import gather_calls, dump_config
from bayou.models.low_level_evidences.node import Node, get_ast_from_json, CHILD_EDGE, SIBLING_EDGE, TooLongLoopingException, TooLongBranchingException


class TooLongPathError(Exception):
    pass


class InvalidSketchError(Exception):
    pass


class Reader():
    def __init__(self, clargs, config, infer=False, dataIsThere=False):
        self.infer = infer
        self.config = config

        if clargs.continue_from is not None or dataIsThere:
            print('Loading Data')
            with open('../low_level_evidences/data/inputs.npy', 'rb') as f:
                self.inputs = pickle.load(f)
            # with open(, 'rb') as f:
            self.nodes = np.load('../low_level_evidences/data/nodes.npy')
            self.edges = np.load('../low_level_evidences/data/edges.npy')


            np.random.seed(0)
            perm = np.random.permutation(len(self.nodes))

            temp_inputs = copy.deepcopy(self.inputs)

            inputs_negative = [input_[perm] for input_ in temp_inputs[:-1]]
            inputs_negative.append([input_surr[perm] for input_surr in temp_inputs[-1][:-1]])
            inputs_negative[-1].append([input_surr_fp[perm] for input_surr_fp in temp_inputs[-1][-1]])

            self.inputs_negative = inputs_negative
            
            jsconfig = dump_config(config)
            with open(os.path.join(clargs.save, 'config.json'), 'w') as f:
                json.dump(jsconfig, fp=f, indent=2)

            if infer:
                self.js_programs = []
                with open('../low_level_evidences/data/js_programs.json', 'rb') as f:
                    for program in ijson.items(f, 'programs.item'):
                        self.js_programs.append(program)
            config.num_batches = 10 #int(len(self.nodes) / config.batch_size)
            print('Done')

        else:
            random.seed(12)
            # read the raw evidences and targets
            print('Reading data file...')
            raw_evidences, raw_targets, js_programs = self.read_data(clargs.input_file[0], infer, save=clargs.save)

            raw_evidences = [[raw_evidence[i] for raw_evidence in raw_evidences] for i, ev in enumerate(config.evidence)]
            raw_evidences[-1] = [[raw_evidence[j] for raw_evidence in raw_evidences[-1]] for j in range(len(config.surrounding_evidence))] # for
            raw_evidences[-1][-1] = [[raw_evidence[j] for raw_evidence in raw_evidences[-1][-1]] for j in range(2)] # is


            config.num_batches = int(len(raw_targets) / config.batch_size)

            ################################

            assert config.num_batches > 0, 'Not enough data'
            sz = config.num_batches * config.batch_size
            for i in range(len(config.evidence)-1): #-1 to leave surrounding evidences
                raw_evidences[i] = raw_evidences[i][:sz]

            for i in range(len(config.surrounding_evidence)-1): #-1 to leave formal params
                raw_evidences[-1][i] = raw_evidences[-1][i][:sz]


            for j in range(2):
                raw_evidences[-1][-1][j] = raw_evidences[-1][-1][j][:sz]

            raw_targets = raw_targets[:sz]
            js_programs = js_programs[:sz]

            # setup input and target chars/vocab
            config.decoder.vocab, config.decoder.vocab_size = self.decoder_api_dict.get_call_dict()
            # adding the same variables for reverse Encoder
            config.reverse_encoder.vocab, config.reverse_encoder.vocab_size = self.decoder_api_dict.get_call_dict()

            # wrangle the evidences and targets into numpy arrays
            self.inputs = [ev.wrangle(data) for ev, data in zip(config.evidence, raw_evidences)]
            self.nodes = np.zeros((sz, config.decoder.max_ast_depth), dtype=np.int32)
            self.edges = np.zeros((sz, config.decoder.max_ast_depth), dtype=np.bool)
            self.targets = np.zeros((sz, config.decoder.max_ast_depth), dtype=np.int32)

            for i, path in enumerate(raw_targets):
                len_path = min(len(path) , config.decoder.max_ast_depth)
                mod_path = path[:len_path]

                self.nodes[i, :len_path]   =  [ p[0] for p in mod_path ]
                self.edges[i, :len_path]   =  [ p[1] for p in mod_path ]
                self.targets[i, :len_path] =  [ p[2] for p in mod_path ]

            self.js_programs = js_programs

            print('Done!')
            # del raw_evidences
            # del raw_targets
            # gc.collect()

            print('Saving...')
            with open('data/inputs.npy', 'wb') as f:
                pickle.dump(self.inputs, f, protocol=4) #pickle.HIGHEST_PROTOCOL)
            # with open(', 'wb') as f:
            np.save('data/nodes', self.nodes)
            np.save('data/edges', self.edges)
            np.save('data/targets', self.targets)

            with open('data/js_programs.json', 'w') as f:
                json.dump({'programs': self.js_programs}, fp=f, indent=2)

            jsconfig = dump_config(config)
            with open(os.path.join(clargs.save, 'config.json'), 'w') as f:
                json.dump(jsconfig, fp=f, indent=2)
            with open('data/config.json', 'w') as f:
                json.dump(jsconfig, fp=f, indent=2)

            print("Saved")
