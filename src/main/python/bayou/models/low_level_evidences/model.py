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

from bayou.models.low_level_evidences.architecture import BayesianEncoder, BayesianDecoder, BayesianReverseEncoder
from bayou.models.low_level_evidences.utils import get_var_list


class Model():
    def __init__(self, config, iterator, infer=False, bayou_mode=True):
        assert config.model == 'lle', 'Trying to load different model implementation: ' + config.model
        self.config = config


        newBatch = iterator.get_next()
        self.prog_ids, self.js_prog_ids, nodes, edges, targets = newBatch[:5]
        ev_data = newBatch[5:]

        nodes = tf.transpose(nodes)
        edges = tf.transpose(edges)

        with tf.variable_scope('Embedding'):
            emb = tf.get_variable('emb', [config.decoder.vocab_size, config.decoder.units])

        with tf.variable_scope("Encoder"):

            self.encoder = BayesianEncoder(config, ev_data, infer or not bayou_mode)
            samples_1 = tf.random_normal([config.batch_size, config.latent_size],
                                       mean=0., stddev=1., dtype=tf.float32)
            self.psi_encoder = self.encoder.psi_mean + tf.sqrt(self.encoder.psi_covariance) * samples_1

        # setup the reverse encoder.
        with tf.variable_scope("Reverse_Encoder"):
            self.reverse_encoder = BayesianReverseEncoder(config, emb, nodes, edges)
            samples_2 = tf.random_normal([config.batch_size, config.latent_size],
                                       mean=0., stddev=1., dtype=tf.float32)
            self.psi_reverse_encoder = self.reverse_encoder.psi_mean + tf.sqrt(self.reverse_encoder.psi_covariance) * samples_2

        # setup the decoder with psi as the initial state
        with tf.variable_scope("Decoder"):
            lift_w = tf.get_variable('lift_w', [config.latent_size, config.decoder.units])
            lift_b = tf.get_variable('lift_b', [config.decoder.units])
            if bayou_mode:
                initial_state = tf.nn.xw_plus_b(self.psi_encoder, lift_w, lift_b, name="Initial_State")
            else:
                initial_state = tf.nn.xw_plus_b(self.psi_reverse_encoder, lift_w, lift_b, name="Initial_State")
            self.decoder = BayesianDecoder(config, emb, initial_state, nodes, edges)


        # get the decoder outputs
        with tf.name_scope("Loss"):
            output = tf.reshape(tf.concat(self.decoder.outputs, 1),
                                [-1, self.decoder.cell1.output_size])
            logits = tf.matmul(output, self.decoder.projection_w) + self.decoder.projection_b
            ln_probs = tf.nn.log_softmax(logits)


            # 1. generation loss: log P(Y | Z)
            cond = tf.not_equal(tf.reduce_sum(self.encoder.psi_mean, axis=1), 0)
            cond = tf.reshape( tf.tile(tf.expand_dims(cond, axis=1) , [1,config.decoder.max_ast_depth]) , [-1] )
            cond =tf.where(cond , tf.ones(cond.shape), tf.zeros(cond.shape))

            self.gen_loss = seq2seq.sequence_loss([logits], [tf.reshape(targets, [-1])],
                                                  [cond])

              # 2. latent loss: negative of the KL-divergence between P(\Psi | f(\Theta)) and P(\Psi)
            self.KL_loss = tf.reduce_mean( 0.5 * tf.reduce_mean( tf.log(self.encoder.psi_covariance) - tf.log(self.reverse_encoder.psi_covariance)
              - 1 + self.reverse_encoder.psi_covariance / self.encoder.psi_covariance
              + tf.square(self.encoder.psi_mean - self.reverse_encoder.psi_mean)/self.encoder.psi_covariance
              , axis=1))

            if bayou_mode:
                self.loss = self.gen_loss #+ 0.001* self.KL_loss
            else:
                self.loss = self.KL_loss

            if infer:
                # self.gen_loss is  P(Y|Z) where Z~P(Z|X)
				# P(Y) = int_Z P(YZ) = int_Z P(Y|Z)P(Z) = int_Z P(Y|Z)P(Z|X)P(Z)/P(Z|X) = sum_Z P(Y|Z)P(Z)/P(Z|X) where Z~P(Z|X)
                # last step by importace_sampling
                # this self.prob_Y is approximate and you need to introduce one more tensor dimension to do this efficiently over multiple trials
				# P(Y) = P(Y|Z)P(Z)/P(Z|X) where Z~P(Z|X)
                self.probY = -1 * self.gen_loss + self.get_multinormal_lnprob(self.psi_encoder) \
                                            - self.get_multinormal_lnprob(self.psi_encoder,self.encoder.psi_mean,self.encoder.psi_covariance)
                self.EncA, self.EncB = self.calculate_ab(self.encoder.psi_mean , self.encoder.psi_covariance)
                self.RevEncA, self.RevEncB = self.calculate_ab(self.reverse_encoder.psi_mean , self.reverse_encoder.psi_covariance)


            self.allEvSigmas = [ ev.sigma for ev in self.config.evidence ]
            #unused if MultiGPU is being used
            with tf.name_scope("train"):
                if bayou_mode:
                    train_ops = get_var_list()['bayou_vars']
                else:
                    train_ops = get_var_list()['rev_encoder_vars']

        if not infer:
            opt = tf.train.AdamOptimizer(config.learning_rate)
            self.train_op = opt.minimize(self.loss, var_list=train_ops)

            var_params = [np.prod([dim.value for dim in var.get_shape()])
                          for var in tf.trainable_variables()]
            print('Model parameters: {}'.format(np.sum(var_params)))


    def get_multinormal_lnprob(self, x, mu=None , Sigma=None ):
        if mu is None:
            mu = tf.zeros(x.shape)
        if Sigma is None:
            Sigma = tf.ones(x.shape)

        # mu is a vector of size [batch_size, latent_size]
        #sigma is another vector of size [batch_size, latent size] denoting a diagonl matrix
        ln_nume =  -0.5 * tf.reduce_sum( tf.square(x-mu) / Sigma, axis=1 )
        ln_deno = self.config.latent_size / 2 * tf.log(2 * np.pi ) + 0.5 * tf.reduce_sum(tf.log(Sigma), axis=1)
        val = ln_nume - ln_deno

        return val

    def calculate_ab(self, mu, Sigma):
        a = -1 /(2*Sigma[:,0]) # slicing a so that a is now of shape (batch_size, 1)
        b = mu / Sigma
        return a, b
