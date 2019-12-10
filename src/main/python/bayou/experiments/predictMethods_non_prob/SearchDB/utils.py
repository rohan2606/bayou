import numpy as np
import pickle
import re



def get_jaccard_distace_api(setA, setB):

    if (len(setA) == 0) and (len(setB) == 0):
        return 1

    setA = set(setA)
    setB = set(setB)

    distance = len(setA & setB) / len(setA | setB)
    return distance



def exact_match_api(setA, setB):

    if len(setA) != len(setB):
        return False

    for item in setA:
        if item not in setB:
         return False
    for item in setB:
        if item not in setA:
         return False

    return True


def exact_match(strA, strB):
    if strB in strA:
        return True
    else:
        return False


def exact_match_ast(dictstrA, strB):

    dictA = eval(dictstrA.replace("u'", "'"))
    dictB = eval(strB)
    if dictA == dictB:
        return True
    else:
        return False


def get_jaccard_distace_seq(dictSeqA, seqB):

    setA = set([tuple(item['calls']) for item in eval(dictSeqA)])
    setB = set([tuple(item['calls']) for item in seqB])

    distance = len(setA & setB) / len(setA | setB)
    return distance

def exact_match_sequence(dictSeqA, seqB):

    setA = [tuple(item['calls']) for item in eval(dictSeqA)]
    setB = [tuple(item['calls']) for item in seqB]
    if len(setA) != len(setB):
        return False

    for item in setA:
      if item not in setB:
         return False
    for item in setB:
      if item not in setA:
         return False
    return True



def get_api_dict():
    #with open("/home/ubuntu/DATABASE/DataWEvidence/outputFiles/dict_api_calls_test.pkl", "rb") as f:
    #        dict_api_calls_test = pickle.load(f)
    #with open("/home/ubuntu/DATABASE/DataWEvidence/outputFiles/dict_api_calls_train.pkl", "rb") as f:
    #        dict_api_calls_train = pickle.load(f)

    with open("/home/ubuntu/DATABASE/DataWEvidence/outputFiles/dict_api_all_data.pkl", "rb") as f:
            dict_api_calls = pickle.load(f)
    #dict_api_calls = dict_api_calls_test
    #dict_api_calls.update(dict_api_calls_train)
    return dict_api_calls

def get_ast_dict():
    with open("/home/ubuntu/DATABASE/DataWEvidence/outputFiles/dict_ast_all_data.pkl", "rb") as f:
        dict_ast = pickle.load(f)
    return dict_ast



def get_sequence_dict():
    with open("/home/ubuntu/DATABASE/DataWEvidence/outputFiles/dict_sequences_all_data.pkl", "rb") as f:
        dict_sequence = pickle.load(f)
    return dict_sequence


def stripJavaDoc(stringBody):
    temp = re.sub(r'/\*\*(.*?)\*\/', '', stringBody.replace('\n','') )
    temp = ' '.join([ word for word in temp.split() if not word.startswith('@') ])
    temp = temp.replace('private', 'public')
    return temp


def get_your_desires(js):
    # get the desired valuess
    desiredBody = stripJavaDoc(js['body'])
    desiredBody = re.sub(r'\*\*(.*?)\*\/', '', desiredBody)
    desireAPIcalls = js['testapicalls']
    desireSeqs = js['testsequences']
    desireAST = str(js['ast'])
    return desiredBody, desireAPIcalls, desireSeqs, desireAST


def rank_statistic(_rank, total, prev_hits, cutoff):
    cutoff = np.array(cutoff)
    hits = prev_hits + (_rank < cutoff)
    prctg = hits / total
    return hits, prctg

def ListToFormattedString(alist, Type):
    # Each item is right-adjusted, width=3
    if Type == 'float':
        formatted_list = ['{:.4f}' for item in alist]
        s = ','.join(formatted_list)
    elif Type == 'int':
        formatted_list = ['{:>3}' for item in alist]
        s = ','.join(formatted_list)
    return s.format(*alist)
