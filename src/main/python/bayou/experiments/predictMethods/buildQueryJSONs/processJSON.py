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
import simplejson as json
import random
from itertools import chain
import re

import bayou.models.low_level_evidences.evidence
from bayou.models.low_level_evidences.utils import gather_calls
from scripts.evidence_extractor import shorten
import scripts.ast_extractor as ast_extractor
import os

max_ast_depth = 32


def processJSONs(inFile, logdir, expNumber=1):
    print("Processing JSONs ... ", end="")
    sys.stdout.flush()
    with open(inFile) as f:
        jsonLines = f.readlines()

    programs = []
    count = 0
    for line in jsonLines:
        line = line.strip()
        if os.path.isfile(line):
            js = processEachJSON(line, expNumber, logdir)

            if js != {}:
                programs.append(js)
                count += 1

    with open(logdir + '/L4TestProgramList.json', 'w') as f:
         json.dump({'programs': programs}, fp=f, indent=2)
    print ("Done")
    return


def processEachJSON(fileName, expNumber, logdir):
    js = extract_evidence(fileName, expNumber)
    js = json.JSONDecoder().decode(js)

    writeFile = fileName.split("/")[-1]

    with open(logdir + '/JSONFiles/' + writeFile, 'w') as f:
        json.dump(js, fp=f, indent=2)

    return js



def modifyInputForExperiment(sample, expNumber):

    if expNumber == 1:

        for ev in ['javaDoc', 'sequences', 'returnType', 'formalParam', 'apicalls', 'types', 'keywords']:
            if ev in sample:
                del sample[ev]

    elif expNumber == 2:
        for ev in ['sequences', 'returnType', 'formalParam', 'apicalls', 'types', 'keywords']:
            if ev in sample:
                del sample[ev]

    elif expNumber == 3:
        for ev in ['sequences', 'apicalls', 'types', 'keywords']:
            if ev in sample:
                del sample[ev]

    elif expNumber == 4:
        for ev in ['sequences']:
            if ev in sample:
                del sample[ev]
        for ev in [ 'apicalls', 'types', 'keywords']:
            sample[ev]



    return sample






def extract_evidence(fileName, expNumber):
    #print('Loading data file...')
    with open(fileName) as f:
        js = json.load(f)
    #print('Done')

    ''' Program_dict dictionary holds Key values in format
    (Key = File_Name Value = dict(Key = String Method_Name, Value = [String ReturnType, List[String] FormalParam , List[String] Sequences] ))
    '''
    programs_dict = dict()

    valid = []
    #This part appends sorrounding evidences

    done = 0
    ignored = 0
    for program in js['programs']:
        try:
            ast_node_graph, ast_paths = ast_extractor.get_ast_paths(program['ast']['_nodes'])
            ast_extractor.validate_sketch_paths(program, ast_paths, max_ast_depth)

            file_name = program['file']
            method_name = program['method']
            returnType = program['returnType'] if 'returnType' in program else "__Constructor__"
            formalParam = program['formalParam'] if 'formalParam' in program else []

            sequences = program['sequences']
            sequences = [[shorten(call) for call in json_seq['calls']] for json_seq in sequences]
            sequences.sort(key=len, reverse=True)


            programs_dict[method_name] = [returnType, formalParam, sequences[0]]
            valid.append(1)


        except (ast_extractor.TooLongPathError, ast_extractor.InvalidSketchError) as e:
            ignored += 1
            valid.append(0)


    choice = None

    if sum(valid) == 0:
        return json.dumps({}, indent=4)
    else:
        while(True):
            rand = random.randint(0, len(valid) - 1)
            if valid[rand] == 1:
                choice = rand
                break



    done = 0
    sample = None
    for pid, program in enumerate(js['programs']):

        if pid != choice:
            continue

        calls = gather_calls(program['ast'])
        apicalls = list(set(chain.from_iterable([bayou.models.low_level_evidences.evidence.APICalls.from_call(call)
                                                 for call in calls])))
        types = list(set(chain.from_iterable([bayou.models.low_level_evidences.evidence.Types.from_call(call)
                                              for call in calls])))
        keywords = list(set(chain.from_iterable([bayou.models.low_level_evidences.evidence.Keywords.from_call(call)
                                                for call in calls])))

        sample = dict(program)
        sample['apicalls'] = apicalls
        sample['types'] = types
        sample['keywords'] = keywords


        sequences = program['sequences']
        sequences = [[shorten(call) for call in json_seq['calls']] for json_seq in sequences]

        sample['sequences'] = sequences[0]

        # Take in classTypes and sample a few
        sample['classTypes'] = list(set(program['classTypes'])) if 'classTypes' in program else []
        if len(sample['classTypes']) == 0:
            del sample['classTypes']

        sample['sorrreturntype'] = []
        sample['sorrformalparam'] = []
        sample['sorrsequences'] = []

        #    (Key = File_Name Value = dict(Key = String Method_Name, Value = [String ReturnType, List[String] FormalParam , List[String] Sequences] ))
        otherMethods = list(programs_dict.keys())

        for method in otherMethods: # Each iterator is a method Name with @linenumber

            # Ignore the current method from list of sorrounding methods
            if method == method_name:
                continue
            # Keep a count on number of sorrounding methods, if it exceeds the random choice, break

            for choice, evidence in zip(programs_dict[method],['sorrreturntype', 'sorrformalparam', 'sorrsequences']):
                sample[evidence].append(choice)

        ## SORR RET
        sample['sorrreturntype'] = list(set(sample['sorrreturntype']))
        if len(sample['sorrreturntype']) == 0:
            del sample['sorrreturntype']

        ## SORR FP
        oldSorrFP = sample['sorrformalparam']
        filteredSorrFP = []
        for FP in oldSorrFP:
            temp = FP
            if len(temp) > 0:
                filteredSorrFP.append( tuple(temp) )

        filteredSorrFP.sort(key=len, reverse=True)
        sample['sorrformalparam'] = list(set(filteredSorrFP))
        if len(sample['sorrformalparam']) == 0:
            del sample['sorrformalparam']

        ## SORR SEQ
        oldSorrSeq = sample['sorrsequences']
        oldSorrSeq.sort(key=len, reverse=True)
        filteredSorrSeq = []
        for seq in oldSorrSeq:
            if len(seq) > 0:
                filteredSorrSeq.append(tuple(seq))

        sample['sorrsequences'] = list(set(filteredSorrSeq))
        if len(sample['sorrsequences']) == 0:
            del sample['sorrsequences']

        done += 1
        # print('Extracted evidence for {} programs'.format(done), end='\n')

        sample = modifyInputForExperiment(sample, expNumber)

    return json.dumps(sample, indent=2)
