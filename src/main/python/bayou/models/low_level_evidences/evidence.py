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

import tensorflow as tf
import numpy as np
import os
import re
import json
import nltk
from itertools import chain
from collections import Counter


from bayou.models.low_level_evidences.utils import CONFIG_ENCODER, CONFIG_INFER, C0, UNK, CHILD_EDGE, SIBLING_EDGE
from bayou.models.low_level_evidences.seqEncoder import seqEncoder
from nltk.stem.wordnet import WordNetLemmatizer


class Evidence(object):

    def init_config(self, evidence, chars_vocab):
        for attr in CONFIG_ENCODER + (CONFIG_INFER if chars_vocab else []):
            self.__setattr__(attr, evidence[attr])



    def dump_config(self):
        js = {attr: self.__getattribute__(attr) for attr in CONFIG_ENCODER + CONFIG_INFER}
        return js

    @staticmethod
    def read_config(js, chars_vocab):
        evidences = []
        for evidence in js:
            name = evidence['name']
            if name == 'apicalls':
                e = APICalls()
            elif name == 'types':
                e = Types()
            elif name == 'keywords':
                e = Keywords()
            elif name == 'callsequences':
                e = CallSequences()
            elif name == 'returntype':
                e = ReturnType()
            elif name == 'classtype':
                e = ClassTypes()
            elif name == 'formalparam':
                e = FormalParam()
            elif name == 'sorrreturntype':
                e = sorrReturnType()
            elif name == 'sorrformalparam':
                e = sorrFormalParam()
            elif name == 'sorrcallsequences':
                e = sorrCallSequences()
            else:
                raise TypeError('Invalid evidence name: {}'.format(name))
            e.name = name
            e.init_config(evidence, chars_vocab)
            evidences.append(e)
        return evidences

    def word2num(self, listOfWords, infer):
        output = []
        for word in listOfWords:
            if word not in self.vocab:
                if not infer:
                    self.vocab[word] = self.vocab_size
                    self.vocab_size += 1
                    output.append(self.vocab[word])
            else:
                output.append(self.vocab[word])
        return output

    def read_data_point(self, program, infer):
        raise NotImplementedError('read_data() has not been implemented')

    def set_chars_vocab(self, data):
        raise NotImplementedError('set_chars_vocab() has not been implemented')

    def wrangle(self, data):
        raise NotImplementedError('wrangle() has not been implemented')

    def placeholder(self, config):
        # type: (object) -> object
        raise NotImplementedError('placeholder() has not been implemented')

    def exists(self, inputs):
        raise NotImplementedError('exists() has not been implemented')

    def init_sigma(self, config):
        raise NotImplementedError('init_sigma() has not been implemented')

    def encode(self, inputs, config):
        raise NotImplementedError('encode() has not been implemented')

    def count_occurence(self, data, f):
        inv_map = {v: k for k, v in self.vocab.items()}
        count = sum([1 if inv_map[val].lower() in f else 0 for arr in arrs for val in arr])
        return count

class Sets(Evidence):


    def wrangle(self, data):
        wrangled = np.zeros((len(data), self.max_nums), dtype=np.int32)
        for i, calls in enumerate(data):
            for j, c in enumerate(calls):
                if j < self.max_nums:
                    wrangled[i, j] = c
        return wrangled

    def placeholder(self, config):
        # type: (object) -> object
        return tf.placeholder(tf.int32, [config.batch_size, self.max_nums])

    def exists(self, inputs):
        return tf.not_equal(tf.reduce_sum(inputs, axis=1), 0)

    def init_sigma(self, config):
        with tf.variable_scope(self.name):
            self.sigma = tf.get_variable('sigma', [])
            self.emb = tf.get_variable('emb', [self.vocab_size, self.units])

    def encode(self, inputs, config):
        with tf.variable_scope(self.name):
            #latent_encoding = tf.zeros([config.batch_size, config.latent_size])
            #inp = tf.slice(inputs, [0, 0, 0], [config.batch_size, 1, self.vocab_size])
            inp = tf.reshape(inputs, [-1])
            emb_inp = tf.nn.embedding_lookup(self.emb, inp)
            encoding = tf.layers.dense(emb_inp, self.units, activation=tf.nn.tanh)
            for i in range(self.num_layers - 1):
                encoding = tf.layers.dense(encoding, self.units, activation=tf.nn.tanh)

            w = tf.get_variable('w', [self.units, config.latent_size])
            b = tf.get_variable('b', [config.latent_size])
            latent_encoding = tf.reshape(tf.nn.xw_plus_b(encoding, w, b), [config.batch_size, self.max_nums, config.latent_size])

            zeros = tf.zeros([config.batch_size, self.max_nums, config.latent_size])
            condition = tf.tile(tf.expand_dims(tf.not_equal(inputs, 0) , axis=2),[1,1,config.latent_size])
            latent_encoding = tf.reduce_sum(tf.where(condition, latent_encoding, zeros), axis=1)

            return latent_encoding



    def f_write(self, data, f):
        f.write('---------------' + self.name +'-------------------------\n')
        arrs = np.squeeze(data) # Now only [self.max_nums]
        inv_map = {v: k for k, v in self.vocab.items()}
        if arrs.shape == ():
            return
        for val in arrs:
            if val == 0:
                continue
            f.write(inv_map[val] + ", ")
        f.write('\n')
        return

# handle sequences as i/p
class Sequences(Evidence):


    def placeholder(self, config):
        # type: (object) -> object
        return tf.placeholder(tf.int32, [config.batch_size, self.max_nums , self.max_depth])

    def wrangle(self, data):
        wrangled = np.zeros((len(data), self.max_nums, self.max_depth), dtype=np.int32)
        for i, seqs in enumerate(data):
            for j, seq in enumerate(seqs):
                if j < self.max_nums : #and seq[0] != 'STOP: # assuming no sequence start with stop and stop has vocab key 0
                    for pos,c in enumerate(seq):
                        if pos < self.max_depth:
                            wrangled[i, j, pos] = c
        return wrangled

    def exists(self, inputs):
        i = tf.reduce_sum(inputs, axis=2)
        return tf.not_equal(tf.reduce_sum(i, axis=1), 0)

    def init_sigma(self, config):
        with tf.variable_scope(self.name):
            self.sigma = tf.get_variable('sigma', [])
            self.emb = tf.get_variable('emb', [self.vocab_size, self.units])

    def encode(self, inputs, config):
        with tf.variable_scope(self.name):
            #latent_encoding = tf.zeros([config.batch_size, config.latent_size])
            #inp = tf.slice(inputs, [0, 0], [config.batch_size, max_depth])
            #can do inversion of input here

            inp = tf.reshape(inputs, [-1, self.max_depth])
            emb_inp = tf.nn.embedding_lookup(self.emb, inp)
            #emb_inp = tf.reverse(emb_inp, axis=[False,True]) # reversed i/p to the encoder

            LSTM_Encoder = seqEncoder(self.num_layers, self.units, emb_inp)
            encoding = LSTM_Encoder.output

            w = tf.get_variable('w', [self.units, config.latent_size])
            b = tf.get_variable('b', [config.latent_size])
            latent_encoding = tf.nn.xw_plus_b(encoding, w, b)

            zeros = tf.zeros([config.batch_size * self.max_nums , config.latent_size])
            latent_encoding = tf.where( tf.not_equal(tf.reduce_sum(inp, axis=1),0),latent_encoding, zeros)

            latent_encoding = tf.reduce_sum(tf.reshape(latent_encoding, [config.batch_size, self.max_nums, config.latent_size]), axis=1)

            return latent_encoding

    def f_write(self, data, f):
        f.write('---------------' + self.name + '-------------------------\n')
        arrs = np.squeeze(data)
        inv_map = {v: k for k, v in self.vocab.items()}
        for arr in arrs:
            if sum(arr)==0:
                continue
            for val in arr:
                if val == 0:
                    continue
                string = inv_map[val]
                f.write(string + ', ')
            f.write('\n')


class APICalls(Sets):

    def __init__(self):
        self.vocab = dict()
        self.vocab['None'] = 0
        self.vocab_size = 1


    def read_data_point(self, program, infer):
        apicalls = program['apicalls'] if 'apicalls' in program else []
        return self.word2num(list(set(apicalls)) , infer)


    @staticmethod
    def from_call(callnode):
        call = callnode['_call']
        call = re.sub('^\$.*\$', '', call)  # get rid of predicates
        name = call.split('(')[0].split('.')[-1]
        name = name.split('<')[0]  # remove generics from call name
        return [name] if name[0].islower() else []  # Java convention

class Types(Sets):

    def __init__(self):
        self.vocab = dict()
        self.vocab['None'] = 0
        self.vocab_size = 1

    def read_data_point(self, program, infer):
        types = program['types'] if 'types' in program else []
        return self.word2num(list(set(types)), infer)

    @staticmethod
    def get_types_re(s):
        patt = re.compile('java[x]?\.(\w*)\.(\w*)(\.([A-Z]\w*))*')
        types = [match.group(4) if match.group(4) is not None else match.group(2)
                 for match in re.finditer(patt, s)]
        primitives = {
            'byte': 'Byte',
            'short': 'Short',
            'int': 'Integer',
            'long': 'Long',
            'float': 'Float',
            'double': 'Double',
            'boolean': 'Boolean',
            'char': 'Character'
        }

        for p in primitives:
            if s == p or re.search('\W{}'.format(p), s):
                types.append(primitives[p])
        return list(set(types))

    @staticmethod
    def from_call(callnode):
        call = callnode['_call']
        types = Types.get_types_re(call)

        if '_throws' in callnode:
            for throw in callnode['_throws']:
                types += Types.get_types_re(throw)

        if '_returns' in callnode:
            types += Types.get_types_re(callnode['_returns'])

        return list(set(types))

class Keywords(Sets):

    def __init__(self):
        nltk.download('wordnet')
        self.lemmatizer = WordNetLemmatizer()
        self.vocab = dict()
        self.vocab['None'] = 0
        self.vocab_size = 1

    STOP_WORDS = {  # CoreNLP English stop words
        "'ll", "'s", "'m", "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between",
        "both", "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
        "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had",
        "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
        "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
        "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me",
        "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only",
        "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
        "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's",
        "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they",
        "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under",
        "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
        "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom",
        "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
        "your", "yours", "yourself", "yourselves", "return", "arent", "cant", "couldnt", "didnt", "doesnt",
        "dont", "hadnt", "hasnt", "havent", "hes", "heres", "hows", "im", "isnt", "its", "lets", "mustnt",
        "shant", "shes", "shouldnt", "thats", "theres", "theyll", "theyre", "theyve", "wasnt", "were",
        "werent", "whats", "whens", "wheres", "whos", "whys", "wont", "wouldnt", "youd", "youll", "youre",
        "youve"
    }

    def lemmatize(self, word):
        w = self.lemmatizer.lemmatize(word, 'v')
        return self.lemmatizer.lemmatize(w, 'n')


    def read_data_point(self, program, infer):
        keywords = [self.lemmatize(k) for k in program['keywords']] if 'keywords' in program else []
        return self.word2num(list(set(keywords)), infer)



    @staticmethod
    def split_camel(s):
        s = re.sub('(.)([A-Z][a-z]+)', r'\1#\2', s)  # UC followed by LC
        s = re.sub('([a-z0-9])([A-Z])', r'\1#\2', s)  # LC followed by UC
        return s.split('#')

    @staticmethod
    def from_call(callnode):
        call = callnode['_call']
        call = re.sub('^\$.*\$', '', call)  # get rid of predicates
        qualified = call.split('(')[0]
        qualified = re.sub('<.*>', '', qualified).split('.')  # remove generics for keywords

        # add qualified names (java, util, xml, etc.), API calls and types
        keywords = list(chain.from_iterable([Keywords.split_camel(s) for s in qualified])) + \
            list(chain.from_iterable([Keywords.split_camel(c) for c in APICalls.from_call(callnode)])) + \
            list(chain.from_iterable([Keywords.split_camel(t) for t in Types.from_call(callnode)]))

        # convert to lower case, omit stop words and take the set
        return list(set([k.lower() for k in keywords if k.lower() not in Keywords.STOP_WORDS]))


class ReturnType(Sets):

    def __init__(self):
        self.vocab = dict()
        self.vocab['None'] = 0
        self.vocab_size = 1


    def read_data_point(self, program, infer):
        returnType = [program['returnType'] if 'returnType' in program else 'NONE']
        return self.word2num(list(set(returnType)), infer)

class ClassTypes(Sets):

    def __init__(self):
        self.vocab = dict()
        self.vocab['None'] = 0
        self.vocab_size = 1

    def read_data_point(self, program, infer):
        classType = program['classTypes'] if 'classTypes' in program else []
        return self.word2num(list(set(classType)), infer)


# handle sequences as i/p
class CallSequences(Sequences):
    def __init__(self):
        self.vocab = dict()
        self.vocab['None'] = 0
        self.vocab_size = 1

    def read_data_point(self, program, infer):
        json_sequences = program['sequences'] if 'sequences' in program else []
        list_seqs = [[]]
        for json_seq in json_sequences:
            tmp_list = json_seq['calls']
            if len(tmp_list) > 1:
                list_seqs.append(self.word2num(tmp_list, infer))
        if len(list_seqs) > 1:
            list_seqs.remove([])
        #return list_seqs
        return list_seqs


    @staticmethod
    def from_call(callnode):
        call = callnode['calls']
        call = re.sub('^\$.*\$', '', call)  # get rid of predicates
        name = call.split('(')[0].split('.')[-1]
        name = name.split('<')[0]  # remove generics from call name
        return [name] if name[0].islower() else []  # Java convention

    def count_occurence(self, data, f):
        count  = 0
        arr = np.squeeze(data)
        inv_map = {v: k for k, v in self.vocab.items()}
        for arr in arrs:
            for val in arr:
                if val != 0:
                    subarr = re.findall(r"[\w']+", inv_map[val].lower()) # excluding java util
                    for sub in subarr:
                        if sub in f:
                            count += 1
        return count


# handle sequences as i/p
class FormalParam(Sequences):

    def __init__(self):
        self.vocab = dict()
        self.vocab['None'] = 0
        self.vocab_size = 1

    def read_data_point(self, program, infer):
        json_sequence = program['formalParam'] if 'formalParam' in program else []
        return [self.word2num(json_sequence, infer)]

    def f_write(self, data, f):
        f.write('---------------' + self.name + '-------------------------\n')
        arr = np.squeeze(data)
        inv_map = {v: k for k, v in self.vocab.items()}

        for val in arr:
            if val == 0:
                continue
            string = inv_map[val]
            f.write(string + ', ')
        f.write('\n')


class sorrCallSequences(Sequences):

    def __init__(self):
        self.vocab = dict()
        self.vocab['None'] = 0
        self.vocab_size = 1

    def read_data_point(self, program, infer):
        json_sequences = program['sorrsequences'] if 'sorrsequences' in program else []
        list_seqs = [[]]
        for i, list_json_seq in enumerate(json_sequences):
            if i> self.max_nums:
                continue
            for json_seq in list_json_seq:
                tmp_list = json_seq['calls']
                if len(tmp_list) > 1:
                    list_seqs.append(self.word2num(tmp_list, infer))
        if len(list_seqs) > 1:
            list_seqs.remove([])
        #return list_seqs
        return list_seqs


# handle sequences as i/p
class sorrFormalParam(Sequences):

    def __init__(self):
        self.vocab = dict()
        self.vocab['None'] = 0
        self.vocab_size = 1

    def read_data_point(self, program, infer):
        json_sequence = program['sorrformalparam'] if 'sorrformalparam' in program else [[]]
        list_seqs = [[]]
        for i, seqs in enumerate(json_sequence):
            if i > self.max_nums:
                continue
            if len(seqs) == 0:
                continue
            list_seqs.append(self.word2num(seqs, infer))
        if len(list_seqs) > 1:
            list_seqs.remove([])
        return list_seqs


class sorrReturnType(Sets):

    def __init__(self):
        self.vocab = dict()
        self.vocab['None'] = 0
        self.vocab_size = 1

    def read_data_point(self, program, infer):
        sorrreturnType = program['sorrreturntype'] if 'sorrreturntype' in program else []
        return self.word2num(list(set(sorrreturnType)), infer)
