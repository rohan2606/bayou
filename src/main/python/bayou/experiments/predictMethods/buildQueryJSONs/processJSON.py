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
from scripts.variable_name_extractor import get_variables
import os

max_ast_depth = 32


def processJSONs(inFile, logdir, config, expNumber=1):
    random.seed(12)
    print("Processing JSONs ... ", end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
          os.makedirs(logdir)

    with open(inFile) as f:
        jsonLines = f.readlines()




    programs = []
    count = 0
    for line in jsonLines:
        line = line.strip()
        if os.path.isfile(line):
            js = processEachJSON(line, expNumber, logdir, config)

            if js != {}:
                programs.append(js)
                count += 1

    with open(logdir + '/L4TestProgramList.json', 'w') as f:
         json.dump({'programs': programs}, fp=f, indent=2)
    print ("Done")
    return count


def processEachJSON(fileName, expNumber, logdir, config):
    js = extract_evidence(fileName, expNumber)
    js = json.JSONDecoder().decode(js)
    js = modifyInputForExperiment(js, expNumber, config)


    #writeFile = fileName.split("/")[-1]
    #with open(logdir + '/JSONFiles/' + writeFile, 'w') as f:
    #    json.dump(js, fp=f, indent=2)

    return js



def stripJavaDoc(stringBody):
    return re.sub(r'/\*\*(.*?)\*\/', '', stringBody.replace('\n',''))

def modifyInputForExperiment(sample, expNumber, config):


    if ( 'apicalls' not in sample ) or ('apicalls' in sample and len(sample['apicalls']) < 1):
         return {}



    sample['testapicalls'] = sample['apicalls']


    ## You need to have all sorrounding infos
    for ev in ['javaDoc', 'Surrounding_Evidences', 'classTypes', 'sequences', 'returnType', 'formalParam', 'apicalls', 'types', 'keywords']:
        if ev not in sample:
            return {}
        if ev == 'javaDoc' and (sample[ev] == None or len(sample[ev].split(" ")) < 1 ):
            return {}
        if ev == 'Surrounding_Evidences' and len(sample[ev]) < 1:
            return {}
        if ev == 'sequences':
            for elem in sample[ev]:
                if elem not in sample['apicalls']:
                    return {}

        if ev == 'returnType' and sample[ev] not in config.evidence[4].vocab:
             return {}

        if ev == 'formalParam':
           for elem in sample[ev]:
               if elem not in config.evidence[5].vocab:
                  return {}



    if expNumber == 0: # onlyJavaDoc

        for ev in [ 'Surrounding_Evidences', 'classTypes', 'sequences', 'returnType', 'formalParam', 'apicalls', 'types', 'keywords']:
            if ev in sample:
                del sample[ev]
        sample['returnType'] = 'None'
        sample['formalParam'] = ['None']


    if expNumber == 1: #only sorrounding infos

        for ev in ['javaDoc', 'sequences', 'returnType', 'formalParam', 'apicalls', 'types', 'keywords']:
            if ev in sample:
                del sample[ev]
        sample['returnType'] = 'None'
        sample['formalParam'] = ['None']

    elif expNumber == 2: # sorrounding plus javadoc


        for ev in [  'sequences', 'returnType', 'formalParam', 'apicalls', 'types', 'keywords']:
            if ev in sample:
                del sample[ev]
        sample['returnType'] = 'None'
        sample['formalParam'] = ['None']


    elif expNumber == 3: ##  sorrounding , ret, fp , jD
        for ev in ['sequences', 'apicalls', 'types', 'keywords']:
            if ev in sample:
                del sample[ev]



    elif expNumber == 4: ## sorrounding plus jD and ret and fp and keywords
        for ev in ['sequences', 'apicalls', 'types']:
            if ev in sample:
                del sample[ev]

    elif expNumber == 5: ## all but sequences
        for ev in ['sequences']:
            if ev in sample:
                del sample[ev]


    elif expNumber == 6: ## all
        for ev in []:
            if ev in sample:
                del sample[ev]

    elif expNumber == 7: ## sequences
        for ev in ['javaDoc', 'Surrounding_Evidences', 'classTypes', 'returnType', 'formalParam', 'apicalls', 'types', 'keywords']:
            if ev in sample:
                del sample[ev]
        sample['returnType'] = 'None'
        sample['formalParam'] = ['None']

    elif expNumber == 8: ## keywords
        for ev in ['javaDoc', 'Surrounding_Evidences', 'classTypes', 'returnType', 'formalParam', 'sequences', 'apicalls', 'types']:
            if ev in sample:
                del sample[ev]
        sample['returnType'] = 'None'
        sample['formalParam'] = ['None']

    elif expNumber == 9: ## apicalls
        for ev in ['javaDoc', 'Surrounding_Evidences', 'classTypes', 'returnType', 'formalParam', 'sequences', 'types', 'keywords']:
            if ev in sample:
                del sample[ev]
        sample['returnType'] = 'None'
        sample['formalParam'] = ['None']

    elif expNumber == 10: ## returnType
        for ev in ['javaDoc', 'Surrounding_Evidences', 'classTypes', 'formalParam', 'sequences', 'apicalls', 'types', 'keywords']:
            if ev in sample:
                del sample[ev]
        sample['formalParam'] = ['None']

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
            sequences = sequences[0]

            header_variable_names, variable_names = get_variables(program['body'])

            programs_dict[method_name] = [returnType, method_name, formalParam, header_variable_names, sequences]
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

        # file_name = program['file']
        # method_name = program['method']

        sample['apicalls'] = apicalls
        sample['types'] = types
        sample['keywords'] = keywords

        sample['body'] = stripJavaDoc(sample['body'])

        method_name = program['method']

        sequences = program['sequences']
        sample['testsequences'] = sequences
        sequences = [[shorten(call) for call in json_seq['calls']] for json_seq in sequences]
        sequences.sort(key=len, reverse=True)
        sample['sequences'] = sequences[0]

        # Take in classTypes and sample a few
        sample['classTypes'] = set(program['classTypes']) if 'classTypes' in program else set()
        sample['classTypes'] = list(sample['classTypes'])

        sample['Surrounding_Evidences']=[]
        #    (Key = File_Name Value = dict(Key = String Method_Name, Value = [String ReturnType, List[String] FormalParam , List[String] Sequences] ))
        otherMethods = list(programs_dict.keys())
        random.shuffle(otherMethods)

        for method in otherMethods: # Each iterator is a method Name with @linenumber

            # Ignore the current method from list of sorrounding methods
            if method == method_name:
                continue
            # Keep a count on number of sorrounding methods, if it exceeds the random choice, break

            methodEvidences={}
            for choice, evidence in zip(programs_dict[method],['surr_returnType', 'surr_methodName', 'surr_formalParam', 'surr_header_vars', 'surr_sequences']):
                if evidence == "surr_returnType":
                    methodEvidences[evidence] = choice
                elif evidence == "surr_formalParam":
                    methodEvidences[evidence] = []
                    for c in choice:
                        methodEvidences[evidence].append('None')
                else:
                    methodEvidences[evidence] = choice

            sample['Surrounding_Evidences'].append(methodEvidences)



        done += 1
        # print('Extracted evidence for {} programs'.format(done), end='\n')


    return json.dumps(sample, indent=2)
