import json
import scripts.ast_extractor as ast_extractor
import bayou.models.low_level_evidences.evidence
from scripts.variable_name_extractor import get_variables
from bayou.models.low_level_evidences.utils import gather_calls
from itertools import chain
from scripts.evidence_extractor import shorten
import re
import random



def stripJavaDoc(stringBody):
    return re.sub(r'/\*\*(.*?)\*\/', '', stringBody.replace('\n',''))

max_ast_depth = 32

def extract_evidence(fileName):
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


    done = 0
    sample = None
    for pid, program in enumerate(js['programs']):

        if '__PDB_FILL__' not in program['body']:
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
                        methodEvidences[evidence].append(c)
                else:
                    methodEvidences[evidence] = choice

            sample['Surrounding_Evidences'].append(methodEvidences)



        done += 1
        # print('Extracted evidence for {} programs'.format(done), end='\n')


    return json.dumps(sample, indent=2)
