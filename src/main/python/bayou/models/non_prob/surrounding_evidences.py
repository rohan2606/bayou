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
import re
from bayou.models.low_level_evidences.utils import CONFIG_ENCODER, CONFIG_INFER
from bayou.models.low_level_evidences.seqEncoder import seqEncoder
from bayou.models.low_level_evidences.seqEncoder_nested import seqEncoder_nested
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

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


    def init_sigma(self, config):
        with tf.variable_scope(self.name):
            self.emb = tf.get_variable('emb', [self.vocab_size, self.units])
            # with tf.variable_scope('global_sigma', reuse=tf.AUTO_REUSE):
            # self.sigma = tf.get_variable('sigma', [])


    def split_words_underscore_plus_camel(self, s):

        # remove unicode
        s = s.encode('ascii', 'ignore').decode('unicode_escape')
        #remove numbers
        s = re.sub(r'\d+', '', s)
        #substitute all non alphabets by # to be splitted later
        s = re.sub("[^a-zA-Z]+", "#", s)
        #camel case split
        s = re.sub('(.)([A-Z][a-z]+)', r'\1#\2', s)  # UC followed by LC
        s = re.sub('([a-z0-9])([A-Z])', r'\1#\2', s)  # LC followed by UC
        vars = s.split('#')

        final_vars = []
        for var in vars:
            var = var.lower()
            w = lemmatizer.lemmatize(var, 'v')
            w = lemmatizer.lemmatize(w, 'n')
            len_w = len(w)
            if len_w > 1 and len_w < 10 :
                final_vars.append(w)
        return final_vars

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
            latent_encoding = tf.reshape(latent_encoding , [config.batch_size * self.max_nums, self.max_depth, config.latent_size])

            count = tf.count_nonzero(tf.reduce_sum(latent_encoding, axis=2), axis=1, dtype=tf.float32)
            latent_encoding = tf.reduce_sum(latent_encoding, axis=1)/(count[:,None]+0.01)

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
            read_programs.append(self.word2num([ret] , infer))
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

    def read_data_point(self, list_of_programs, infer):
        read_programs = []
        for program in list_of_programs:
            seq = program['surr_sequences'] if 'surr_sequences' in program else []
            read_programs.append(self.word2num(seq , infer))
        return read_programs


class surr_methodName(SetsOfSequences):

    def __init__(self):
        self.vocab = dict()
        self.vocab['None'] = 0
        self.vocab_size = 1

    def read_data_point(self, list_of_programs, infer):
        read_programs = []
        for program in list_of_programs:
            methodName = program['surr_methodName'] if 'surr_methodName' in program else []
            methodName = methodName.split('@')[0]
            tokens = self.split_words_underscore_plus_camel(methodName)
            read_programs.append(self.word2num(tokens , infer))

        return read_programs # number_of_surr_methods * lenght_of_methodName(4)



class surr_formalParam(SetsOfSomething):

    def __init__(self):
        self.vocab = [dict() , dict()]
        self.vocab[0]['None'] = 0
        self.vocab[1]['None'] = 0
        self.vocab_size = [1,1]


    def word2num(self, listOfWords, id, infer):
        output = []
        for word in listOfWords:
            if word not in self.vocab[id]:
                if not infer:
                    self.vocab[id][word] = self.vocab_size[id]
                    self.vocab_size[id] += 1
                    output.append(self.vocab[id][word])
            else:
                output.append(self.vocab[id][word])
                # with open("/home/ubuntu/evidences_used.txt", "a") as f:
                #      f.write('Evidence Type :: ' + self.name + " , " + "Evidence Value :: " + word + "\n")

        return output


    def read_data_point(self, list_of_programs, infer):
        surr_formals = []
        surr_formals_vars = []
        for program in list_of_programs:
            fp_types = program['surr_formalParam'] if 'surr_formalParam' in program else []
            surr_formals.append(self.word2num(fp_types, 0, infer))

            list_of_var_name_ids = []
            varNames = program['surr_header_vars'] if 'surr_header_vars' in program else []
            for varName in varNames:
                tokens = self.split_words_underscore_plus_camel(varName)
                var_name_ids = self.word2num(tokens, 1,infer)
                list_of_var_name_ids.append(var_name_ids)
            surr_formals_vars.append(list_of_var_name_ids)


        return [surr_formals , surr_formals_vars] # [number_of_surr_methods * length_of_formal_params , number_of_surr_methods *  length_of_formal_params * length_of_Vars]

    def wrangle(self, data):
        data_0 = data[0]
        data_1 = data[1]
        wrangled_0 = np.zeros((len(data_0), self.max_nums, self.max_depth), dtype=np.int32)
        for i, method in enumerate(data_0):
            for j, keyword in enumerate(method):
                if j < self.max_nums:
                    for k,call in enumerate(keyword):
                        if k < self.max_depth:
                            wrangled_0[i, j, k] = call

        wrangled_1 = np.zeros((len(data_1), self.max_nums, self.max_depth, 3), dtype=np.int32)
        for i, method in enumerate(data_1):
            for j, keyword in enumerate(method):
                if j < self.max_nums:
                    for k, var_name_tokens in enumerate(keyword):
                        if k < self.max_depth:
                            for l, token in enumerate(var_name_tokens):
                                if l < 3:
                                    wrangled_1[i, j, k, l] = token

        return [wrangled_0, wrangled_1]


    def placeholder(self, config):
        # type: (object) -> object
        return [tf.placeholder(tf.int32, [config.batch_size, self.max_nums, self.max_depth]) , tf.placeholder(tf.int32, [config.batch_size, self.max_nums, self.max_depth, 3])]


    def exists(self, inputs, config, infer):
        ii = tf.expand_dims(tf.reduce_sum(inputs[0], axis=[1,2]),axis=1)
        ij = tf.expand_dims(tf.reduce_sum(inputs[1], axis=[1,2,3]),axis=1)
        i = ii + ij
        # Drop a few types of evidences during training
        if not infer:
            i_shaped_zeros = tf.zeros_like(i)
            rand = tf.random_uniform( (config.batch_size,1) )
            i = tf.where(tf.less(rand, self.ev_drop_prob) , i, i_shaped_zeros)
        i = tf.reduce_sum(i, axis=1)

        return i #tf.not_equal(i, 0) # [batch_size]

    def init_sigma(self, config):
        with tf.variable_scope(self.name):
            self.emb = [None, None]
            self.emb[0] = tf.get_variable('emb0', [self.vocab_size[0], self.units])
            self.emb[1] = tf.get_variable('emb1', [self.vocab_size[1], self.units])

    def encode(self, inputs, config, infer):
        with tf.variable_scope(self.name):

            # Drop some inputs
            inputs_0 = tf.reshape(inputs[0], [config.batch_size * self.max_nums, self.max_depth])
            inputs_1 = tf.reshape(inputs[1], [config.batch_size * self.max_nums * self.max_depth, 3])

            if not infer:
                inp_shaped_zeros = tf.zeros_like(inputs_0)
                rand = tf.random_uniform( (config.batch_size * self.max_nums, self.max_depth  ) )
                inputs_0 = tf.where(tf.less(rand, self.ev_call_drop_prob) , inputs_0, inp_shaped_zeros)

                inp_shaped_zeros = tf.zeros_like(inputs_1)
                rand = tf.random_uniform( (config.batch_size * self.max_nums * self.max_depth, 3  ) )
                inputs_1 = tf.where(tf.less(rand, self.ev_call_drop_prob) , inputs_1, inp_shaped_zeros)


            LSTM_Encoder = seqEncoder(self.num_layers, self.units, inputs_1, config.batch_size * self.max_nums * self.max_depth, self.emb[1], config.latent_size)
            encoding = LSTM_Encoder.output

            w = tf.get_variable('w0', [self.units, config.latent_size ])
            b = tf.get_variable('b0', [config.latent_size])
            latent_encoding_variables_intermediate = tf.nn.xw_plus_b(encoding, w, b)

            zeros = tf.zeros_like(latent_encoding_variables_intermediate)
            latent_encoding_variables_intermediate = tf.where( tf.not_equal(tf.reduce_sum(inputs_1, axis=1),0),latent_encoding_variables_intermediate, zeros)
            latent_encoding_variables_intermediate = tf.reshape(latent_encoding_variables_intermediate, [config.batch_size * self.max_nums, self.max_depth, -1])


            input_vars_mod_cond = tf.reduce_sum(tf.reshape(inputs_1 , [config.batch_size * self.max_nums , self.max_depth, 3]), axis=2)

            LSTM_Encoder = seqEncoder_nested(self.num_layers, self.units, inputs_0, config.batch_size * self.max_nums, self.emb[0], latent_encoding_variables_intermediate, input_vars_mod_cond)
            encoding = LSTM_Encoder.output

            w = tf.get_variable('w1', [self.units, config.latent_size ])
            b = tf.get_variable('b1', [config.latent_size])
            latent_encoding = tf.nn.xw_plus_b(encoding, w, b)

            zeros = tf.zeros_like(latent_encoding)
            cond_0 = tf.equal(tf.reduce_sum(inputs_0, axis=1), 0)
            cond_1 = tf.equal(tf.reduce_sum(input_vars_mod_cond, axis=1), 0)
            cond = tf.logical_and(cond_0 , cond_1)

            latent_encoding = tf.where( cond, zeros, latent_encoding)

            latent_encoding = tf.reshape(latent_encoding, [config.batch_size ,self.max_nums, config.latent_size])

            return latent_encoding
