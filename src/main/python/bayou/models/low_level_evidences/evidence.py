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

import gensim
from bayou.models.low_level_evidences.utils import CONFIG_ENCODER, CONFIG_INFER
from bayou.models.low_level_evidences.seqEncoder import seqEncoder
from bayou.models.low_level_evidences.biRNN import biRNN

from nltk.stem.wordnet import WordNetLemmatizer
import wordninja

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
            elif name == 'javadoc':
                e = JavaDoc()
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
                # with open("/home/ubuntu/evidences_used.txt", "a") as f:
                #      f.write('Evidence Type :: ' + self.name + " , " + "Evidence Value :: " + word + "\n")

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

    def exists(self, inputs, config, infer):
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

    def exists(self, inputs, config, infer):
        i = tf.expand_dims(tf.reduce_sum(inputs, axis=1),axis=1)
        # Drop a few types of evidences during training
        if not infer:
            i_shaped_zeros = tf.zeros_like(i)
            rand = tf.random_uniform( (config.batch_size,1) )
            i = tf.where(tf.less(rand, self.ev_drop_prob) , i, i_shaped_zeros)

        i = tf.reduce_sum(i, axis=1)

        return tf.not_equal(i, 0)

    def init_sigma(self, config):
        with tf.variable_scope(self.name):
            self.emb = tf.get_variable('emb', [self.vocab_size, self.units])
        # with tf.variable_scope('global_sigma', reuse=tf.AUTO_REUSE):
            self.sigma = tf.get_variable('sigma', [])

    def encode(self, inputs, config, infer):
        with tf.variable_scope(self.name):

            # Drop some inputs
            if not infer:
                inp_shaped_zeros = tf.zeros_like(inputs)
                rand = tf.random_uniform( (config.batch_size, self.max_nums) )
                inputs = tf.where(tf.less(rand, self.ev_call_drop_prob) , inputs, inp_shaped_zeros)

            inputs = tf.reshape(inputs, [-1])

            emb_inp = tf.nn.embedding_lookup(self.emb, inputs)
            encoding = tf.layers.dense(emb_inp, self.units, activation=tf.nn.tanh)
            for i in range(self.num_layers - 1):
                encoding = tf.layers.dense(encoding, self.units, activation=tf.nn.tanh)

            w = tf.get_variable('w', [self.units, config.latent_size])
            b = tf.get_variable('b', [config.latent_size])
            latent_encoding = tf.nn.xw_plus_b(encoding, w, b)

            zeros = tf.zeros([config.batch_size * self.max_nums, config.latent_size])
            condition = tf.not_equal(inputs, 0)

            latent_encoding = tf.where(condition, latent_encoding, zeros)
            latent_encoding = tf.reduce_sum(tf.reshape(latent_encoding , [config.batch_size, self.max_nums, config.latent_size]), axis=1)
            return latent_encoding

# handle sequences as i/p
class Sequences(Evidence):


    def placeholder(self, config):
        # type: (object) -> object
        return tf.placeholder(tf.int32, [config.batch_size, self.max_depth])

    def wrangle(self, data):
        wrangled = np.zeros((len(data), self.max_depth), dtype=np.int32)
        for i, seqs in enumerate(data):
            seq = seqs[0]# NOT A BUG every sequence is read as List of List
            for pos,c in enumerate(seq):
                if pos < self.max_depth and c != 0:
                    wrangled[i, self.max_depth - 1 - pos] = c
        return wrangled

    def exists(self, inputs, config, infer):
        i = tf.expand_dims(tf.reduce_sum(inputs, axis=1),axis=1)
        # Drop a few types of evidences during training
        if not infer:
            i_shaped_zeros = tf.zeros_like(i)
            rand = tf.random_uniform( (config.batch_size,1) )
            i = tf.where(tf.less(rand, self.ev_drop_prob) , i, i_shaped_zeros)
        i = tf.reduce_sum(i, axis=1)

        return tf.not_equal(i, 0)


    def init_sigma(self, config):
        with tf.variable_scope(self.name):
            self.emb = tf.get_variable('emb', [self.vocab_size, self.units])
        # with tf.variable_scope('global_sigma', reuse=tf.AUTO_REUSE):
            self.sigma = tf.get_variable('sigma', [])

    def encode(self, inputs, config, infer):
        with tf.variable_scope(self.name):
            # Drop some inputs
            if not infer:
                inp_shaped_zeros = tf.zeros_like(inputs)
                rand = tf.random_uniform( (config.batch_size, self.max_depth) )
                inputs = tf.where(tf.less(rand, self.ev_call_drop_prob) , inputs, inp_shaped_zeros)

            LSTM_Encoder = seqEncoder(self.num_layers, self.units, inputs, config.batch_size, self.emb, config.latent_size)
            encoding = LSTM_Encoder.output

            w = tf.get_variable('w', [self.units, config.latent_size ])
            b = tf.get_variable('b', [config.latent_size])
            latent_encoding = tf.nn.xw_plus_b(encoding, w, b)

            zeros = tf.zeros([config.batch_size , config.latent_size])
            latent_encoding = tf.where( tf.not_equal(tf.reduce_sum(inputs, axis=1),0),latent_encoding, zeros)

            return latent_encoding



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
        returnType = [program['returnType'] if 'returnType' in program else '__Constructor__']

        return self.word2num(returnType , infer)

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
        json_seq = program['sequences'] if 'sequences' in program else []
        return [self.word2num(json_seq, infer)]


    @staticmethod
    def from_call(callnode):
        call = callnode['calls']
        call = re.sub('^\$.*\$', '', call)  # get rid of predicates
        name = call.split('(')[0].split('.')[-1]
        name = name.split('<')[0]  # remove generics from call name
        return [name] if name[0].islower() else []  # Java convention



# handle sequences as i/p
class FormalParam(Sequences):

    def __init__(self):
        self.vocab = dict()
        self.vocab['None'] = 0
        self.vocab_size = 1

    def read_data_point(self, program, infer):
        json_sequence = program['formalParam'] if 'formalParam' in program else []
        if 'None' not in json_sequence:
            json_sequence.insert(0, 'Start')
            json_sequence.insert(0, 'None')
        return [self.word2num(json_sequence, infer)]


# handle sequences as i/p
class JavaDoc(Sequences):

    def __init__(self):
        self.vocab = dict()
        self.vocab['None'] = 0
        self.vocab_size = 1
        self.word2vecModel = gensim.models.KeyedVectors.load_word2vec_format('/home/ubuntu/GoogleNews-vectors-negative300.bin', binary=True)
        self.n_Dims=300
        self.lemmatizer = WordNetLemmatizer()



    def read_data_point(self, program, infer):

        string_sequence = program['javaDoc'] if ('javaDoc' in program and program['javaDoc'] is not None) else []
        if len(string_sequence) == 0:
             return [[]]

        javadoc_list = string_sequence.strip().split()
        # replace all non alphabetical char into underscore
        javadoc_list = [re.sub("[^a-zA-Z]", '_', w) for w in javadoc_list]

            # break the terms using underscores
        tmp_list = []
        for t in javadoc_list:
               s = re.split("_+", t)
               tmp_list.extend(s)

        result_list = []
        for x in tmp_list:
            x = x.lower()
                # x = spell(x)

            x = wordninja.split(x)
            for word in x:
                y = self.lemmatizer.lemmatize(word, 'v')
                y = self.lemmatizer.lemmatize(y, 'n')
                if len(y) > 1:
                    result_list.append(y)

        return [self.word2num(result_list , infer)]

    def init_sigma(self, config):
        with tf.variable_scope(self.name):
            #REPLACE BY WORD2VEC
            # self.emb = tf.get_variable('emb', [self.vocab_size, self.units])

            vecrep_words = np.zeros((self.vocab_size,self.n_Dims), dtype=np.float32)
            for key in self.vocab:
            	vocab_ind = self.vocab[key]
            	if key in self.word2vecModel:
            		vecrep_words[vocab_ind] = self.word2vecModel[key]

            self.emb = tf.Variable(vecrep_words, name='emb',trainable=True)
        # with tf.variable_scope('global_sigma', reuse=tf.AUTO_REUSE):
            #self.sigma = tf.Variable(0.10, name='sigma', trainable=True) #tf.get_variable('sigma', [])
            self.sigma = tf.get_variable('sigma', [])


    def encode(self, inputs, config, infer):
        with tf.variable_scope(self.name):
            # Drop some inputs
            if not infer:
                inp_shaped_zeros = tf.zeros_like(inputs)
                rand = tf.random_uniform( (config.batch_size, self.max_depth) )
                inputs = tf.where(tf.less(rand, self.ev_call_drop_prob) , inputs, inp_shaped_zeros)

            BiGRU_Encoder = biRNN(self.num_layers, self.units, inputs, config.batch_size, self.emb, config.latent_size)
            encoding = BiGRU_Encoder.output

            w = tf.get_variable('w', [self.units, config.latent_size ])
            b = tf.get_variable('b', [config.latent_size])
            latent_encoding = tf.nn.xw_plus_b(encoding, w, b)

            zeros = tf.zeros([config.batch_size , config.latent_size])
            latent_encoding = tf.where( tf.not_equal(tf.reduce_sum(inputs, axis=1),0),latent_encoding, zeros)

            return latent_encoding

class sorrReturnType(Sets):

    def __init__(self):
        self.vocab = dict()
        self.vocab['None'] = 0
        self.vocab_size = 1

    def read_data_point(self, program, infer):
        sorrreturnType = program['sorrreturntype'] if 'sorrreturntype' in program else []
        return self.word2num(sorrreturnType, infer)




# handle sequences as i/p
class SetsOfSequences(Evidence):


    def placeholder(self, config):
        # type: (object) -> object
        return tf.placeholder(tf.int32, [config.batch_size, self.max_nums, self.max_depth])

    def wrangle(self, data):
        wrangled = np.zeros((len(data), self.max_nums, self.max_depth), dtype=np.int32)
        for i, seqs in enumerate(data):
            for j, seq in enumerate(seqs):
                if j < self.max_nums:
                    for pos,c in enumerate(seq):
                        if pos < self.max_depth and c != 0:
                            wrangled[i, j, self.max_depth - 1 - pos] = c
        return wrangled

    def exists(self, inputs, config, infer):
        i = tf.expand_dims(tf.reduce_sum(inputs, axis=[1,2]),axis=1)
        # Drop a few types of evidences during training
        if not infer:
            i_shaped_zeros = tf.zeros_like(i)
            rand = tf.random_uniform( (config.batch_size,1) )
            i = tf.where(tf.less(rand, self.ev_drop_prob) , i, i_shaped_zeros)
        i = tf.reduce_sum(i, axis=1)

        return tf.not_equal(i, 0)


    def init_sigma(self, config):
        with tf.variable_scope(self.name):
            self.emb = tf.get_variable('emb', [self.vocab_size, self.units])
        # with tf.variable_scope('global_sigma', reuse=tf.AUTO_REUSE):
            self.sigma = tf.get_variable('sigma', [])


    def encode(self, inputs, config, infer):
        with tf.variable_scope(self.name):
            # Drop some inputs
            inputs = tf.reshape(inputs, [config.batch_size * self.max_nums, self.max_depth])

            if not infer:
                inp_shaped_zeros = tf.zeros_like(inputs)
                rand = tf.random_uniform( (config.batch_size * self.max_nums, self.max_depth  ) )
                inputs = tf.where(tf.less(rand, self.ev_call_drop_prob) , inputs, inp_shaped_zeros)


            LSTM_Encoder = seqEncoder(self.num_layers, self.units, inputs, config.batch_size * self.max_nums, self.emb, config.latent_size)
            encoding = LSTM_Encoder.output

            w = tf.get_variable('w', [self.units, config.latent_size ])
            b = tf.get_variable('b', [config.latent_size])
            latent_encoding = tf.nn.xw_plus_b(encoding, w, b)

            zeros = tf.zeros_like(latent_encoding)
            latent_encoding = tf.where( tf.not_equal(tf.reduce_sum(inputs, axis=1),0),latent_encoding, zeros)

            latent_encoding = tf.reduce_sum( tf.reshape(latent_encoding, [config.batch_size ,self.max_nums, config.latent_size]) , axis=1)

            return latent_encoding


class sorrCallSequences(SetsOfSequences):

    def __init__(self):
        self.vocab = dict()
        self.vocab['None'] = 0
        self.vocab_size = 1

    def read_data_point(self, program, infer):
        json_sequences = program['sorrsequences'] if 'sorrsequences' in program else [[]]
        list_seqs = [[]]
        for json_seq in json_sequences:
            list_seqs.append(self.word2num(json_seq, infer))
        if len(list_seqs) > 1:
            list_seqs.remove([])

        return list_seqs


# handle sequences as i/p
class sorrFormalParam(SetsOfSequences):

    def __init__(self):
        self.vocab = dict()
        self.vocab['None'] = 0
        self.vocab_size = 1

    def read_data_point(self, program, infer):
        json_sequence = program['sorrformalparam'] if 'sorrformalparam' in program else [[]]
        list_seqs = [[]]
        for i, seqs in enumerate(json_sequence):
            if i > self.max_nums or len(seqs) == 0 :
                continue
            list_seqs.append(self.word2num(seqs, infer))
        if len(list_seqs) > 1:
            list_seqs.remove([])
        return list_seqs
