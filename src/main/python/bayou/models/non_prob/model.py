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
from tensorflow.contrib import legacy_seq2seq as seq2seq
import numpy as np

from bayou.models.non_prob.architecture import BayesianEncoder, BayesianReverseEncoder
from bayou.models.low_level_evidences.utils import get_var_list


class Model():
    def __init__(self, config, iterator, infer=False, bayou_mode=True):
        assert config.model == 'lle', 'Trying to load different model implementation: ' + config.model
        self.config = config


        newBatch = iterator.get_next()
        nodes, edges, nodes_negative, edges_negative = newBatch[:4]
        ev_data = newBatch[4:14]
        surr_input = newBatch[14:17]
        surr_input_fp = newBatch[17:]


        self.nodes = tf.transpose(nodes)
        self.edges = tf.transpose(edges)

        self.nodes_negative = tf.transpose(nodes_negative)
        self.edges_negative = tf.transpose(edges_negative)

        with tf.variable_scope("Encoder"):
            self.encoder = BayesianEncoder(config, ev_data, surr_input, surr_input_fp, infer)
            self.psi_encoder = self.encoder.psi_mean

        # setup the reverse encoder.
        with tf.variable_scope("Reverse_Encoder", reuse=tf.AUTO_REUSE):
            embAPI = tf.get_variable('embAPI', [config.reverse_encoder.vocab_size, config.reverse_encoder.units])
            embRT = tf.get_variable('embRT', [config.evidence[4].vocab_size, config.reverse_encoder.units])
            embFS = tf.get_variable('embFS', [config.evidence[5].vocab_size, config.reverse_encoder.units])
            self.reverse_encoder = BayesianReverseEncoder(config, embAPI, self.nodes, self.edges,  ev_data[4], embRT, ev_data[5], embFS)
            self.psi_reverse_encoder = self.reverse_encoder.psi_mean

            self.reverse_encoder_negative = BayesianReverseEncoder(config, embAPI, self.nodes_negative, self.edges_negative, ev_data[4], embRT, ev_data[5], embFS)
            self.psi_reverse_encoder_negative = self.reverse_encoder_negative.psi_mean

            self.loss = tf.reduce_sum(tf.math.maximum(0., 0.05 - self.cosine_similarity(self.psi_encoder, self.psi_reverse_encoder) + self.cosine_similarity(self.psi_encoder, self.psi_reverse_encoder_negative)), axis=0)

            #unused if MultiGPU is being used
            with tf.name_scope("train"):
                train_ops = get_var_list()['all_vars']

        if not infer:
            opt = tf.train.AdamOptimizer(config.learning_rate)
            self.train_op = opt.minimize(self.loss, var_list=train_ops)

            var_params = [np.prod([dim.value for dim in var.get_shape()])
                          for var in tf.trainable_variables()]
            print('Model parameters: {}'.format(np.sum(var_params)))



    def cosine_similarity(self, a, b):
       norm_a = tf.nn.l2_normalize(a,1)
       norm_b = tf.nn.l2_normalize(b,1)
       batch_similarity = tf.reduce_sum(tf.multiply(norm_a, norm_b), axis=1)
       #batch_loss = 1 - batch_similarity
       #total_similarity = tf.reduce_mean(batch_similarity, axis=0)
       return batch_similarity
