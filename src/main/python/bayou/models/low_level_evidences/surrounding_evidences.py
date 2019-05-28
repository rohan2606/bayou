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
import json

from bayou.models.low_level_evidences.utils import CONFIG_ENCODER, CONFIG_INFER

from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

class SurroundingEvidence(object):

    def __init__(self):
        self.vocab = None
        self.vocab_size = 0

    def read_data_point(self, program, infer):
        list_of_programs = program['Surrounding_Evidences'] if 'Surrounding_Evidences' in program else []
        # print(list_of_programs)
        data = [ev.read_data_point(list_of_programs, infer) for ev in self.internal_evidences] #self.config.surrounding_evidence]
        return data


    def wrangle(self, data):
        wrangled = [ev.wrangle(ev_data) for ev, ev_data in zip(self.internal_evidences , data )]

        return wrangled

    def placeholder(self, config):
        # type: (object) -> object
        return [ev.placeholder(config) for ev in config.surrounding_evidence]


    def exists(self, inputs, config, infer):

        temp = [ev.exists(input, config, infer) for input, ev in zip(inputs, config.surrounding_evidence)]
        temp = tf.reduce_sum(tf.stack(temp, 0),0)
        return tf.not_equal(temp, 0)

    def init_sigma(self, config):
        with tf.variable_scope(self.name):
            self.emb = tf.get_variable('emb', [self.vocab_size, self.units])
        # with tf.variable_scope('global_sigma', reuse=tf.AUTO_REUSE):
            self.sigma = tf.get_variable('sigma', [])
            [ev.init_sigma(config) for ev in config.surrounding_evidence]


    def encode(self, inputs, config, infer):
        with tf.variable_scope(self.name):
            encodings = [ev.encode(i, config, infer) for ev, i in zip(config.surrounding_evidence, inputs)]
            #number_of_ev * batch_size * number_of_methods * latent_size
            encodings = tf.reduce_mean(tf.stack(encodings, axis=0), axis=0)
            #batch_size * number_of_methods * latent_size
            encodings = tf.reduce_sum(encodings, axis=1)
        return encodings



    def init_config(self, evidence, chars_vocab):
        for attr in CONFIG_ENCODER + (CONFIG_INFER if chars_vocab else []):
            self.__setattr__(attr, evidence[attr])

    def dump_config(self):
        js = {attr: self.__getattribute__(attr) for attr in CONFIG_ENCODER + CONFIG_INFER}
        js['evidence'] = [ev.dump_config() for ev in self.internal_evidences]
        return js


    # @staticmethod
    def read_config(self, js, chars_vocab):
        evidences = []
        for evidence in js:
            name = evidence['name']
            if name == 'surr_sequences':
                e = surr_sequences()
            elif name == 'surr_methodName':
                e = surr_methodName()
            elif name == 'surr_header_vars':
                e = surr_header_vars()
            elif name == 'surr_returnType':
                e = surr_returnType()
            else:
                raise TypeError('Invalid evidence name: {}'.format(name))
            e.name = name
            e.init_config(evidence, chars_vocab)
            evidences.append(e)
        self.internal_evidences = evidences

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


# handle sequences as i/p
class SetsOfSomething(object):

    def init_config(self, evidence, chars_vocab):
        for attr in CONFIG_ENCODER + (CONFIG_INFER if chars_vocab else []):
            self.__setattr__(attr, evidence[attr])

    def dump_config(self):
        js = {attr: self.__getattribute__(attr) for attr in CONFIG_ENCODER + CONFIG_INFER}
        return js

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

    def placeholder(self, config):
        # type: (object) -> object
        return tf.placeholder(tf.int32, [config.batch_size, self.max_nums, self.max_depth])

    def wrangle(self, data):
        wrangled = np.zeros((len(data), self.max_nums, self.max_depth), dtype=np.int32)
        for i, method in enumerate(data):
            for j, keyword in enumerate(method):
                if j < self.max_nums:
                    for k,call in enumerate(keyword):
                        if k < self.max_depth:
                            wrangled[i, j, k] = call
        return wrangled

    def exists(self, inputs, config, infer):
        i = tf.expand_dims(tf.reduce_sum(inputs, axis=[1,2]),axis=1)
        # Drop a few types of evidences during training
        if not infer:
            i_shaped_zeros = tf.zeros_like(i)
            rand = tf.random_uniform( (config.batch_size,1) )
            i = tf.where(tf.less(rand, self.ev_drop_prob) , i, i_shaped_zeros)
        i = tf.reduce_sum(i, axis=1)

        return i #tf.not_equal(i, 0) # [batch_size]


    def init_sigma(self, config):
        with tf.variable_scope(self.name):
            self.emb = tf.get_variable('emb', [self.vocab_size, self.units])
        # with tf.variable_scope('global_sigma', reuse=tf.AUTO_REUSE):
            self.sigma = tf.get_variable('sigma', [])





class SetsOfSets(SetsOfSomething):

    def encode(self, inputs, config, infer):
        with tf.variable_scope(self.name):
            # Drop some inputs
            inputs = tf.reshape(inputs, [config.batch_size * self.max_nums, self.max_depth])

            if not infer:
                inp_shaped_zeros = tf.zeros_like(inputs)
                rand = tf.random_uniform( (config.batch_size * self.max_nums, self.max_depth  ) )
                inputs = tf.where(tf.less(rand, self.ev_call_drop_prob) , inputs, inp_shaped_zeros)

            inputs = tf.reshape(inputs, [-1])

            emb_inp = tf.nn.embedding_lookup(self.emb, inputs)
            encoding = tf.layers.dense(emb_inp, self.units, activation=tf.nn.tanh)
            for i in range(self.num_layers - 1):
                encoding = tf.layers.dense(encoding, self.units, activation=tf.nn.tanh)

            w = tf.get_variable('w', [self.units, config.latent_size])
            b = tf.get_variable('b', [config.latent_size])
            latent_encoding = tf.nn.xw_plus_b(encoding, w, b)

            zeros = tf.zeros([config.batch_size * self.max_nums * self.max_depth, config.latent_size])
            condition = tf.not_equal(inputs, 0)

            latent_encoding = tf.where(condition, latent_encoding, zeros)
            latent_encoding = tf.reduce_sum(tf.reshape(latent_encoding , [config.batch_size * self.max_nums, self.max_depth, config.latent_size]), axis=1)

            latent_encoding = tf.reshape(latent_encoding, [config.batch_size ,self.max_nums, config.latent_size])
            return latent_encoding


class surr_returnType(SetsOfSets):

    def __init__(self):
        self.vocab = dict()
        self.vocab['None'] = 0
        self.vocab_size = 1


    def read_data_point(self, list_of_programs, infer):
        read_programs = []
        for program in list_of_programs:
            ret = program['surr_returnType'] if 'surr_returnType' in program else []
            read_programs.append(self.word2num(list(set(ret)) , infer))
        return read_programs



# handle sequences as i/p
class SetsOfSequences(SetsOfSomething):

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

            latent_encoding = tf.reshape(latent_encoding, [config.batch_size ,self.max_nums, config.latent_size])

            return latent_encoding


class surr_sequences(SetsOfSequences):

    def __init__(self):
        self.vocab = dict()
        self.vocab['None'] = 0
        self.vocab_size = 1

    def read_data_point(self, program, infer):
        read_programs = []
        for program in list_of_programs:
            seq = program['surr_sequences'] if 'surr_sequences' in program else []
            read_programs.append(self.word2num(list(set(seq)) , infer))
        return read_programs


class surr_methodName(SetsOfSequences):

    def __init__(self):
        self.vocab = dict()
        self.vocab['None'] = 0
        self.vocab_size = 1

    def read_data_point(self, program, infer):
        read_programs = []
        for program in list_of_programs:
            met = program['surr_methodName'] if 'surr_methodName' in program else []
            read_programs.append(self.word2num(list(set(met)) , infer))
        return read_programs



class surr_formalParam(SetsOfSequences):

    def __init__(self):
        self.vocab = dict()
        self.vocab['None'] = 0
        self.vocab_size = 1

    def read_data_point(self, program, infer):
        read_programs = []
        for program in list_of_programs:
            fp = program['surr_formalParam'] if 'surr_formalParam' in program else []
            read_programs.append(self.word2num(list(set(fp)) , infer))
        return read_programs



class surr_header_vars(SetsOfSequences):

    def __init__(self):
        self.vocab = dict()
        self.vocab['None'] = 0
        self.vocab_size = 1

    def read_data_point(self, program, infer):
        read_programs = []
        for program in list_of_programs:
            met = program['surr_header_vars'] if 'surr_header_vars' in program else []
            read_programs.append(self.word2num(list(set(met)) , infer))
        return read_programs
