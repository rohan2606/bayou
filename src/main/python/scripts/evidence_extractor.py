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
import argparse
import sys
import json
import ijson.backends.yajl2_cffi as ijson
import math
import random
import numpy as np
from itertools import chain
import re

import bayou.models.low_level_evidences.evidence
from bayou.models.low_level_evidences.utils import gather_calls
import ast_extractor

HELP = """Use this script to extract evidences from a raw data file with sequences generated by driver.
You can also filter programs based on number and length of sequences, and control the samples from each program."""


def shorten(call):
    call = re.sub('^\$.*\$', '', call)  # get rid of predicates
    name = call.split('(')[0].split('.')[-1]
    name = name.split('<')[0]  # remove generics from call name
    return name

def extract_evidence(clargs):
    print('Loading data file...')

    f = open(clargs.input_file[0] , 'rb')
    print('Done')
    done = 0
    programs = []

    ''' Program_dict dictionary holds Key values in format
    (Key = File_Name Value = dict(Key = String Method_Name, Value = [String ReturnType, List[String] FormalParam , List[String] Sequences] ))
    '''
    programs_dict = dict()


    returnDict = dict()
    FP_Dict = dict()

    valid = []
    #This part appends sorrounding evidences
    done = 0
    ignored = 0
    for program in ijson.items(f, 'programs.item'):
        if 'ast' not in program:
            continue
        try:
            ast_node_graph, ast_paths = ast_extractor.get_ast_paths(program['ast']['_nodes'])
            ast_extractor.validate_sketch_paths(program, ast_paths, clargs.max_ast_depth)

            file_name = program['file']
            method_name = program['method']

            sequences = program['sequences']
            sequences = [[shorten(call) for call in json_seq['calls']] for json_seq in sequences]
            sequences.sort(key=len, reverse=True)

            if 'returnType' not in program:
                continue

            if program['returnType'] == 'None':
                program['returnType'] = '__Constructor__'

            returnType = program['returnType']

            if returnType not in returnDict:
                returnDict[returnType] = 1
            else:
                returnDict[returnType] += 1

            formalParam = program['formalParam'] if 'formalParam' in program else []

            for type in formalParam:
                if type not in FP_Dict:
                    FP_Dict[type] = 1
                else:
                    FP_Dict[type] += 1

            # if len(sequences) > clargs.max_seqs or (len(sequences) == 1 and len(sequences[0]['calls']) == 1) or \
            #     any([len(sequence['calls']) > clargs.max_seq_length for sequence in sequences]):
            #         raise ast_extractor.TooLongPathError


            if file_name not in programs_dict:
                programs_dict[file_name] = dict()

            if method_name in programs_dict[file_name]:
                print('Hit Found')

            programs_dict[file_name][method_name] = [returnType, formalParam, sequences[0]]


        except (ast_extractor.TooLongPathError, ast_extractor.InvalidSketchError) as e:
            ignored += 1

        done += 1
        if done % 100000 == 0:
            print('Extracted evidences of sorrounding features for {} programs'.format(done), end='\n')

    print('')

    print('{:8d} programs/asts in training data'.format(done))
    print('{:8d} programs/asts ignored by given config'.format(ignored))
    print('{:8d} programs/asts to search over'.format(done - ignored))


    topRetKeys = dict()
    for w in sorted(returnDict, key=returnDict.get, reverse=True)[:1000]:
        topRetKeys[w] = returnDict[w]

    topFPKeys = dict()
    for w in sorted(FP_Dict, key=FP_Dict.get, reverse=True)[:1000]:
        topFPKeys[w] = FP_Dict[w]

    f.close()
    f = open(clargs.input_file[0] , 'rb')
    done = 0
    for program in ijson.items(f, 'programs.item'):
        if 'ast' not in program:
            continue
        try:
            ast_node_graph, ast_paths = ast_extractor.get_ast_paths(program['ast']['_nodes'])
            ast_extractor.validate_sketch_paths(program, ast_paths, clargs.max_ast_depth)

            file_name = program['file']
            method_name = program['method']

            sequences = program['sequences']
            sequences = [[shorten(call) for call in json_seq['calls']] for json_seq in sequences]
            sequences.sort(key=len, reverse=True)
            program['sequences'] = sequences

            if 'returnType' not in program:
                continue
            if program['returnType'] == 'None':
                program['returnType'] = '__Constructor__'

            if program['returnType'] not in topRetKeys:
                program['returnType'] = '__UDT__'

            returnType = program['returnType']

            formalParam = program['formalParam'] if 'formalParam' in program else []
            newFP = []
            for type in formalParam:
                if type not in topFPKeys:
                    type = '__UDT__'
                newFP.append(type)




            # if len(sequences) > clargs.max_seqs or (len(sequences) == 1 and len(sequences[0]['calls']) == 1) or \
            #         any([len(sequence['calls']) > clargs.max_seq_length for sequence in sequences]):
            #     continue

            sample = dict(program)
            calls = gather_calls(program['ast'])
            apicalls = list(set(chain.from_iterable([bayou.models.low_level_evidences.evidence.APICalls.from_call(call)
                                                     for call in calls])))
            types = list(set(chain.from_iterable([bayou.models.low_level_evidences.evidence.Types.from_call(call)
                                                  for call in calls])))
            keywords = list(set(chain.from_iterable([bayou.models.low_level_evidences.evidence.Keywords.from_call(call)
                                                    for call in calls])))
            random.shuffle(apicalls)
            random.shuffle(types)
            random.shuffle(keywords)
            sample['apicalls'] = apicalls
            sample['types'] = types
            sample['keywords'] = keywords
            sample['returnType'] = returnType
            sample['formalParam'] = newFP

            classTypes = list(set(program['classTypes'])) if 'classTypes' in program else []
            random.shuffle(classTypes)
            sample['classTypes'] = classTypes

            sample['sorrreturntype'] = []
            sample['sorrformalparam'] = []
            sample['sorrsequences'] = []



            #(Key = File_Name Value = dict(Key = String Method_Name, Value = [String ReturnType, List[String] FormalParam , List[String] Sequences] ))

            otherMethods = list(programs_dict[file_name].keys())
            random.shuffle(otherMethods)

            for method in otherMethods: # Each iterator is a method Name with @linenumber
                # Ignore the current method from list of sorrounding methods
                if method == method_name:
                    continue

                for choice, evidence in zip(programs_dict[file_name][method],['sorrreturntype', 'sorrformalparam', 'sorrsequences']):
                    sample[evidence].append(choice)

            programs.append(sample)

        except (ast_extractor.TooLongPathError, ast_extractor.InvalidSketchError) as e:
            ignored += 1

        done += 1
        if done % 100000 == 0:
            print('Extracted evidence [API/Type/Keywords/Sorrounding Evidences] for {} programs'.format(done), end='\n')

    random.shuffle(programs)


    print('\nWriting to {}...'.format(clargs.output_file[0]), end='')
    with open(clargs.output_file[0], 'w') as f:
        json.dump({'programs': programs}, fp=f, indent=2)
    print('done')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=HELP)
    parser.add_argument('input_file', type=str, nargs=1,
                        help='input data file')
    parser.add_argument('output_file', type=str, nargs=1,
                        help='output data file')
    parser.add_argument('--python_recursion_limit', type=int, default=10000,
                        help='set recursion limit for the Python interpreter')
    parser.add_argument('--max_ast_depth', type=int, default=32,
                        help='max ast depth for out program ')


    clargs = parser.parse_args()
    sys.setrecursionlimit(clargs.python_recursion_limit)
    extract_evidence(clargs)
