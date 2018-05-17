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
from itertools import chain
from collections import Counter

from bayou.models.low_level_evidences.utils import CONFIG_ENCODER, CONFIG_INFER, C0, UNK, CHILD_EDGE, SIBLING_EDGE
from bayou.models.low_level_evidences.seqEncoder import seqEncoder
from bayou.models.low_level_evidences.gru_tree import TreeEncoder
from bayou.models.low_level_evidences.node import Node

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
            elif name == 'sequences':
                e = Sequences()
            elif name == 'ast':
                e = ast()
            else:
                raise TypeError('Invalid evidence name: {}'.format(name))
            e.init_config(evidence, chars_vocab)
            evidences.append(e)
        return evidences

    def read_data_point(self, program):
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

    def evidence_loss(self, psi, encoding, config):
        raise NotImplementedError('evidence_loss() has not been implemented')

    def split(self, data, num_batches, axis=0):
        return np.split(data, num_batches, axis)



class APICalls(Evidence):

    def read_data_point(self, program):
        apicalls = program['apicalls'] if 'apicalls' in program else []
        return list(set(apicalls))

    def set_chars_vocab(self, data):
        counts = Counter([c for apicalls in data for c in apicalls])
        self.chars = sorted(counts.keys(), key=lambda w: counts[w], reverse=True)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.vocab_size = len(self.vocab)

    def wrangle(self, data):
        wrangled = np.zeros((len(data), 1, self.vocab_size), dtype=np.int32)
        for i, apicalls in enumerate(data):
            for c in apicalls:
                if c in self.vocab:
                    wrangled[i, 0, self.vocab[c]] = 1
        return wrangled

    def placeholder(self, config):
        # type: (object) -> object
        return tf.placeholder(tf.float32, [config.batch_size, 1, self.vocab_size])

    def exists(self, inputs):
        i = tf.reduce_sum(inputs, axis=2)
        return tf.not_equal(tf.count_nonzero(i, axis=1), 0)

    def init_sigma(self, config):
        with tf.variable_scope('apicalls'):
            self.sigma = tf.get_variable('sigma', [])

    def encode(self, inputs, config):
        with tf.variable_scope('apicalls'):
            latent_encoding = tf.zeros([config.batch_size, config.latent_size])
            inp = tf.slice(inputs, [0, 0, 0], [config.batch_size, 1, self.vocab_size])
            inp = tf.reshape(inp, [-1, self.vocab_size])
            encoding = tf.layers.dense(inp, self.units, activation=tf.nn.tanh)
            for i in range(self.num_layers - 1):
                encoding = tf.layers.dense(encoding, self.units, activation=tf.nn.tanh)
            w = tf.get_variable('w', [self.units, config.latent_size])
            b = tf.get_variable('b', [config.latent_size])
            latent_encoding += tf.nn.xw_plus_b(encoding, w, b)
            return latent_encoding

    def evidence_loss(self, psi, encoding, config):
        sigma_sq = tf.square(self.sigma)
        loss = 0.5 * (config.latent_size * tf.log(2 * np.pi * sigma_sq + 1e-10)
                      + tf.square(encoding - psi) / sigma_sq)
        return loss

    @staticmethod
    def from_call(callnode):
        call = callnode['_call']
        call = re.sub('^\$.*\$', '', call)  # get rid of predicates
        name = call.split('(')[0].split('.')[-1]
        name = name.split('<')[0]  # remove generics from call name
        return [name] if name[0].islower() else []  # Java convention

    def print_ev(self, data):
        print('---------------APICalls---Used-------------------------\n')
        arr = np.squeeze(data)
        assert(len(list(arr.shape)) == 1)
        inv_map = {v: k for k, v in self.vocab.items()}
        for j, val in enumerate(arr):
            if val == 1: #because ev_data is wrangled
                print(inv_map[j])

class Types(Evidence):

    def read_data_point(self, program):
        types = program['types'] if 'types' in program else []
        return list(set(types))

    def set_chars_vocab(self, data):
        counts = Counter([t for types in data for t in types])
        self.chars = sorted(counts.keys(), key=lambda w: counts[w], reverse=True)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.vocab_size = len(self.vocab)

    def wrangle(self, data):
        wrangled = np.zeros((len(data), 1, self.vocab_size), dtype=np.int32)
        for i, types in enumerate(data):
            for t in types:
                if t in self.vocab:
                    wrangled[i, 0, self.vocab[t]] = 1
        return wrangled

    def placeholder(self, config):
        # type: (object) -> object
        return tf.placeholder(tf.float32, [config.batch_size, 1, self.vocab_size])

    def exists(self, inputs):
        i = tf.reduce_sum(inputs, axis=2)
        return tf.not_equal(tf.count_nonzero(i, axis=1), 0)

    def init_sigma(self, config):
        with tf.variable_scope('types'):
            self.sigma = tf.get_variable('sigma', [])

    def encode(self, inputs, config):
        with tf.variable_scope('types'):
            latent_encoding = tf.zeros([config.batch_size, config.latent_size])
            inp = tf.slice(inputs, [0, 0, 0], [config.batch_size, 1, self.vocab_size])
            inp = tf.reshape(inp, [-1, self.vocab_size])
            encoding = tf.layers.dense(inp, self.units, activation=tf.nn.tanh)
            for i in range(self.num_layers - 1):
                encoding = tf.layers.dense(encoding, self.units, activation=tf.nn.tanh)
            w = tf.get_variable('w', [self.units, config.latent_size])
            b = tf.get_variable('b', [config.latent_size])
            latent_encoding += tf.nn.xw_plus_b(encoding, w, b)
            return latent_encoding

    def evidence_loss(self, psi, encoding, config):
        sigma_sq = tf.square(self.sigma)
        loss = 0.5 * (config.latent_size * tf.log(2 * np.pi * sigma_sq + 1e-10)
                      + tf.square(encoding - psi) / sigma_sq)
        return loss


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

    def print_ev(self, data):
        arr = np.squeeze(data)
        assert(len(list(arr.shape)) == 1)
        inv_map = {v: k for k, v in self.vocab.items()}
        for j, val in enumerate(arr):
            if val == 1: #because ev_data is wrangled
                print(inv_map[j])

class Keywords(Evidence):

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

    def read_data_point(self, program):
        keywords = program['keywords'] if 'keywords' in program else []
        return list(set(keywords))

    def set_chars_vocab(self, data):
        counts = Counter([c for keywords in data for c in keywords])
        self.chars = sorted(counts.keys(), key=lambda w: counts[w], reverse=True)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.vocab_size = len(self.vocab)

    def wrangle(self, data):
        wrangled = np.zeros((len(data), 1, self.vocab_size), dtype=np.int32)
        for i, keywords in enumerate(data):
            for k in keywords:
                if k in self.vocab and k not in Keywords.STOP_WORDS:
                    wrangled[i, 0, self.vocab[k]] = 1
        return wrangled

    def placeholder(self, config):
        # type: (object) -> object
        return tf.placeholder(tf.float32, [config.batch_size, 1, self.vocab_size])

    def exists(self, inputs):
        i = tf.reduce_sum(inputs, axis=2)
        return tf.not_equal(tf.count_nonzero(i, axis=1), 0)

    def init_sigma(self, config):
        with tf.variable_scope('keywords'):
            self.sigma = tf.get_variable('sigma', [])

    def encode(self, inputs, config):
        with tf.variable_scope('keywords'):
            latent_encoding = tf.zeros([config.batch_size, config.latent_size])
            inp = tf.slice(inputs, [0, 0, 0], [config.batch_size, 1, self.vocab_size])
            inp = tf.reshape(inp, [-1, self.vocab_size])
            encoding = tf.layers.dense(inp, self.units, activation=tf.nn.tanh)
            for i in range(self.num_layers - 1):
                encoding = tf.layers.dense(encoding, self.units, activation=tf.nn.tanh)
            w = tf.get_variable('w', [self.units, config.latent_size])
            b = tf.get_variable('b', [config.latent_size])
            latent_encoding += tf.nn.xw_plus_b(encoding, w, b)
            return latent_encoding

    def evidence_loss(self, psi, encoding, config):
        sigma_sq = tf.square(self.sigma)
        loss = 0.5 * (config.latent_size * tf.log(2 * np.pi * sigma_sq + 1e-10)
                      + tf.square(encoding - psi) / sigma_sq)
        return loss

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

    def print_ev(self, data):
        print('-----------------------Keywords--Used-------------------\n')
        arr = np.squeeze(data)
        assert(len(list(arr.shape)) == 1)
        inv_map = {v: k for k, v in self.vocab.items()}
        for j, val in enumerate(arr):
            if val == 1: #because ev_data is wrangled
                print(inv_map[j])

# TODO: handle Javadoc with word2vec
class Javadoc(Evidence):

    def read_data_point(self, program, infer=False):
        javadoc = program['javadoc'] if 'javadoc' in program else None
        if not javadoc:
            javadoc = UNK
        try:  # do not consider non-ASCII javadoc
            javadoc.encode('ascii')
        except UnicodeEncodeError:
            javadoc = UNK
        javadoc = javadoc.split()
        return javadoc

    def set_dicts(self, data):
        if self.pretrained_embed:
            save_dir = os.path.join(self.save_dir, 'embed_' + self.name)
            with open(os.path.join(save_dir, 'config.json')) as f:
                js = json.load(f)
            self.chars = js['chars']
        else:
            self.chars = [C0] + list(set([w for point in data for w in point]))
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.vocab_size = len(self.vocab)

    def print_ev(self):
        print('--------------------------Javadoc--Used--------------------\n')

# handle sequences as i/p
class Sequences(Evidence):

    def read_data_point(self, program):
        json_sequences = program['sequences'] if 'sequences' in program else []
        list_seqs = [[]]
        for json_seq in json_sequences:
            tmp_list = json_seq['calls']
            if len(tmp_list) > 1:
                list_seqs.append(tmp_list)
        if len(list_seqs) > 1:
            list_seqs.remove([])
        #return list_seqs
        return list_seqs

    def set_chars_vocab(self, data):
        counts = Counter([c for seq in data for c in seq])
        self.chars = sorted(counts.keys(), key=lambda w: counts[w], reverse=True)
        self.chars.insert(0,'STOP')
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.vocab_size = len(self.vocab)

    def wrangle(self, data):
        with tf.variable_scope("sequences"):
            max_length = self.tile
            wrangled = np.zeros((len(data), max_length), dtype=np.int32)
            for i, seq in enumerate(data):
                for pos,c in enumerate(seq):
                    if c in self.vocab and pos < max_length:
                        wrangled[i, pos] = self.vocab[c]
        return wrangled

    def placeholder(self, config):
        # type: (object) -> object
        max_length = self.tile
        return tf.placeholder(tf.int32, [config.batch_size, max_length])

    def exists(self, inputs):
        i = tf.reduce_sum(inputs, axis=1)
        i = tf.expand_dims(i,axis=1)
        return tf.not_equal(tf.count_nonzero(i, axis=1), 0)


    def init_sigma(self, config):
        with tf.variable_scope('sequences'):
            self.sigma = tf.get_variable('sigma', [])
            self.emb = tf.get_variable('emb', [self.vocab_size, self.units])


    def encode(self, inputs, config):
        with tf.variable_scope('sequences'):
            latent_encoding = tf.zeros([config.batch_size, config.latent_size])
            max_length = self.tile
            inp = tf.slice(inputs, [0, 0], [config.batch_size, max_length])
            #can do inversion of input here
            emb_inp = tf.nn.embedding_lookup(self.emb, inp)
            emb_inp = tf.reverse(emb_inp, axis=[False,True]) # reversed i/p to the encoder

            LSTM_Encoder = seqEncoder(self.num_layers, self.units, emb_inp)
            encoding = LSTM_Encoder.output

            w = tf.get_variable('w', [self.units, config.latent_size])
            b = tf.get_variable('b', [config.latent_size])
            latent_encoding += tf.nn.xw_plus_b(encoding, w, b)

            return latent_encoding

    def evidence_loss(self, psi, encoding, config):
        sigma_sq = tf.square(self.sigma)
        loss = 0.5 * (config.latent_size * tf.log(2 * np.pi * sigma_sq + 1e-10)
                      + tf.square(encoding - psi) / sigma_sq)
        return loss

    @staticmethod
    def from_call(callnode):
        call = callnode['calls']
        call = re.sub('^\$.*\$', '', call)  # get rid of predicates
        name = call.split('(')[0].split('.')[-1]
        name = name.split('<')[0]  # remove generics from call name
        return [name] if name[0].islower() else []  # Java convention

    def print_ev(self, data):
        print('---------------Sequences--Used-------------------------\n')
        arr = np.squeeze(data)
        inv_map = {v: k for k, v in self.vocab.items()}
        for val in arr:
            string = inv_map[val]
            if string == 'STOP':
                print('' , end='')
            else:
                print(string , end=',')
        print()


class ast(Evidence):

    def read_data_point(self, program):
        return []

    def set_chars_vocab(self, data):

        counts = Counter([n for path in data for (n, _) in path])
        counts[C0] = 1
        self.chars = sorted(counts.keys(), key=lambda w: counts[w], reverse=True)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.vocab_size = len(self.vocab)

    def wrangle(self, data):
        with tf.variable_scope("ast"):
            max_ast_length = self.tile
            #In the 3rd dimension, the 0-th col is node and 1-st col is edge
            nodes_edges = np.zeros((len(data), max_ast_length, 2), dtype=np.int32)
            #edges = np.zeros((len(data), max_ast_length), dtype=np.bool)
            for i, path in enumerate(data):
                nodes_edges[i, :len(path), 0] = list(map(self.vocab.get, [p[0] for p in path]))
                nodes_edges[i, :len(path), 1] = [p[1] == CHILD_EDGE for p in path]

        return nodes_edges

    '''def split(self, data, num_batches, axis=0):
        nodes_edges = data
        nodes = np.split(nodes, num_batches, axis)
        edges = np.split(edges, num_batches, axis)
        zipper = [(nodes[i], edges[i]) for i in range(num_batches)]
        return zipper'''


    def placeholder(self, config):
        # type: (object) -> object
        max_ast_length = self.tile
        nodes_edges = tf.placeholder(tf.int32, [config.batch_size, max_ast_length, 2])
        #edges = tf.placeholder(tf.bool, [config.batch_size, max_ast_length])
        return nodes_edges

    def exists(self, inputs):
        return True


    def init_sigma(self, config):
        with tf.variable_scope('ast'):
            self.sigma = tf.get_variable('sigma', [])
            self.emb = tf.get_variable('emb', [self.vocab_size, self.units])


    def encode(self, inputs, config):
        with tf.variable_scope('ast'):
            latent_encoding = tf.zeros([config.batch_size, config.latent_size])
            max_ast_depth = self.tile

            nodes_edges = inputs
            nodes, edges = tf.unstack(nodes_edges, axis=2)
            nodes = tf.unstack(nodes, axis=1)
            edges = tf.cast(tf.unstack(edges, axis=1), dtype=tf.bool)


            Path_encoder = TreeEncoder(self.emb, config.batch_size, nodes, edges,self.num_layers, \
                                self.units, max_ast_depth, self.units)

            encoding = Path_encoder.last_output

            w = tf.get_variable('w', [self.units, config.latent_size])
            b = tf.get_variable('b', [config.latent_size])
            latent_encoding += tf.nn.xw_plus_b(encoding, w, b)

            return latent_encoding

    def evidence_loss(self, psi, encoding, config):
        sigma_sq = tf.square(self.sigma)
        loss = 0.5 * (config.latent_size * tf.log(2 * np.pi * sigma_sq + 1e-10)
                      + tf.square(encoding - psi) / sigma_sq)
        return loss

    def print_ev(self, data):
        print('---------------Ast--Used-------------------------\n')
        arr = np.squeeze(data)
        max_length = self.tile
        inv_map = {v: k for k, v in self.vocab.items()}
        for i in range(max_length):
            string = inv_map[arr[i][0]]
            if string == 'DSubTree':
                print('' , end='')
            else:
                print(string , end=',')
                print(arr[i][1])
