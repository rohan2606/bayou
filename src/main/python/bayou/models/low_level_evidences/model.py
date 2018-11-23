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

from bayou.models.low_level_evidences.architecture import BayesianEncoder, BayesianDecoder, BayesianReverseEncoder, SimpleDecoder
from bayou.models.low_level_evidences.utils import get_var_list


class Model():
    def __init__(self, config, iterator, infer=False, bayou_mode=True):
        assert config.model == 'lle', 'Trying to load different model implementation: ' + config.model
        self.config = config


        newBatch = iterator.get_next()
        self.prog_ids, self.js_prog_ids, nodes, edges, targets, tree_nodes, tree_edges = newBatch[:7]
        ev_data = newBatch[7:]

        nodes = tf.transpose(nodes)
        edges = tf.transpose(edges)

        #tree nodes are batch_size * 5 * max_ast_depth
        tree_nodes = tf.transpose(tf.reshape(tree_nodes, [-1, config.decoder.max_ast_depth]))
        tree_edges = tf.transpose(tf.reshape(tree_edges, [-1, config.decoder.max_ast_depth]))

        with tf.variable_scope("Embedding"):
            emb = tf.get_variable('emb', [config.decoder.vocab_size, config.decoder.units])

        with tf.variable_scope("Encoder"):
            self.encoder = BayesianEncoder(config, ev_data, infer or not bayou_mode)
            samples_1 = tf.random_normal([config.batch_size, config.latent_size],
                                       mean=0., stddev=1., dtype=tf.float32)
            self.psi_encoder = self.encoder.psi_mean + tf.sqrt(self.encoder.psi_covariance) * samples_1

        # setup the reverse encoder.
        with tf.variable_scope("Reverse_Encoder"):
            self.reverse_encoder = BayesianReverseEncoder(config, emb, tree_nodes, tree_edges, ev_data[4], config.evidence[4].emb, ev_data[5], config.evidence[5].emb)
            samples_2 = tf.random_normal([config.batch_size, config.latent_size],
                                       mean=0., stddev=1., dtype=tf.float32)
            self.psi_reverse_encoder = self.reverse_encoder.psi_mean + tf.sqrt(self.reverse_encoder.psi_covariance) * samples_2

        # setup the decoder with psi as the initial state
        with tf.variable_scope("Decoder"):
            lift_w = tf.get_variable('lift_w', [config.latent_size, config.decoder.units])
            lift_b = tf.get_variable('lift_b', [config.decoder.units])


            if bayou_mode or infer:
                initial_state = tf.nn.xw_plus_b(self.psi_encoder, lift_w, lift_b, name="Initial_State")
            else:
                initial_state = tf.nn.xw_plus_b(self.psi_reverse_encoder, lift_w, lift_b, name="Initial_State")

            self.decoder = BayesianDecoder(config, emb, initial_state, nodes, edges)

        with tf.variable_scope("RE_Decoder"):
            ## RE

            emb_RE = config.evidence[4].emb * 0.0 #tf.get_variable('emb_RE', [config.evidence[4].vocab_size, config.evidence[4].units])

            lift_w_RE = tf.get_variable('lift_w_RE', [config.latent_size, config.evidence[4].units])
            lift_b_RE = tf.get_variable('lift_b_RE', [config.evidence[4].units])

            if bayou_mode or infer:
                initial_state_RE = tf.nn.xw_plus_b(self.psi_encoder, lift_w_RE, lift_b_RE, name="Initial_State_RE")
            else:
                initial_state_RE = tf.nn.xw_plus_b(self.psi_reverse_encoder, lift_w_RE, lift_b_RE, name="Initial_State_RE")

            input_RE = tf.transpose(tf.reverse_v2(tf.zeros_like(ev_data[4]), axis=[1]))
            output = SimpleDecoder(config, emb_RE, initial_state_RE, input_RE, config.evidence[4])

            projection_w_RE = tf.get_variable('projection_w_RE', [config.evidence[4].units, config.evidence[4].vocab_size])
            projection_b_RE = tf.get_variable('projection_b_RE', [config.evidence[4].vocab_size])
            logits_RE = tf.nn.xw_plus_b(output.outputs[-1] , projection_w_RE, projection_b_RE)

            labels_RE = tf.one_hot(tf.squeeze(tf.zeros_like(ev_data[4])) , config.evidence[4].vocab_size , dtype=tf.int32)
            loss_RE = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_RE, logits=logits_RE)

            cond = tf.not_equal(tf.reduce_sum(self.encoder.psi_mean, axis=1), 0)
            # cond = tf.reshape( tf.tile(tf.expand_dims(cond, axis=1) , [1,config.evidence[5].max_depth]) , [-1] )
            self.loss_RE = tf.reduce_mean(tf.where(cond , loss_RE, tf.zeros(cond.shape)))

        with tf.variable_scope("FS_Decoder"):
            #FS
            emb_FS = config.evidence[5].emb #tf.get_variable('emb_FS', [config.evidence[5].vocab_size, config.evidence[5].units])
            lift_w_FS = tf.get_variable('lift_w_FS', [config.latent_size, config.evidence[5].units])
            lift_b_FS = tf.get_variable('lift_b_FS', [config.evidence[5].units])

            if bayou_mode or infer:
                initial_state_FS = tf.nn.xw_plus_b(self.psi_encoder, lift_w_FS, lift_b_FS, name="Initial_State_FS")
            else:
                initial_state_FS = tf.nn.xw_plus_b(self.psi_reverse_encoder, lift_w_FS, lift_b_FS, name="Initial_State_FS")

            input_FS = tf.transpose(tf.reverse_v2(ev_data[5], axis=[1]))
            self.decoder_FS = SimpleDecoder(config, emb_FS, initial_state_FS, input_FS, config.evidence[5])

            output = tf.reshape(tf.concat(self.decoder_FS.outputs, 1), [-1, self.decoder_FS.cell1.output_size])
            logits_FS = tf.matmul(output, self.decoder_FS.projection_w_FS) + self.decoder_FS.projection_b_FS


            # logits_FS = output
            targets_FS = tf.reverse_v2(tf.concat( [ tf.zeros_like(ev_data[5][:,-1:]) , ev_data[5][:, :-1]], axis=1) , axis=[1])


            # self.gen_loss_FS = tf.contrib.seq2seq.sequence_loss(logits_FS, target_FS,
            #                                       tf.ones_like(target_FS, dtype=tf.float32))
            cond = tf.not_equal(tf.reduce_sum(self.encoder.psi_mean, axis=1), 0)
            cond = tf.reshape( tf.tile(tf.expand_dims(cond, axis=1) , [1,config.evidence[5].max_depth]) , [-1] )
            cond =tf.where(cond , tf.ones(cond.shape), tf.zeros(cond.shape))


            self.gen_loss_FS = seq2seq.sequence_loss([logits_FS], [tf.reshape(targets_FS, [-1])],
                                                  [cond])

        # get the decoder outputs
        with tf.name_scope("Loss"):
            output = tf.reshape(tf.concat(self.decoder.outputs, 1),
                                [-1, self.decoder.cell1.output_size])
            logits = tf.matmul(output, self.decoder.projection_w) + self.decoder.projection_b
            ln_probs = tf.nn.log_softmax(logits)


            # 1. generation loss: log P(Y | Z)
            cond = tf.not_equal(tf.reduce_sum(self.encoder.psi_mean, axis=1), 0)
            cond = tf.reshape( tf.tile(tf.expand_dims(cond, axis=1) , [1,config.decoder.max_ast_depth]) , [-1] )
            cond = tf.where(cond , tf.ones(cond.shape), tf.zeros(cond.shape))



            self.gen_loss = seq2seq.sequence_loss([logits], [tf.reshape(targets, [-1])],
                                                  [cond])

              # 2. latent loss: negative of the KL-divergence between P(\Psi | f(\Theta)) and P(\Psi)
            KL_loss = 0.5 * tf.reduce_mean( tf.log(self.encoder.psi_covariance) - tf.log(self.reverse_encoder.psi_covariance)
              - 1 + self.reverse_encoder.psi_covariance / self.encoder.psi_covariance
              + tf.square(self.encoder.psi_mean - self.reverse_encoder.psi_mean)/self.encoder.psi_covariance
              , axis=1)



            KL_cond = tf.not_equal(tf.reduce_sum(self.encoder.psi_mean, axis=1) , 0)
            self.KL_loss = tf.reduce_mean( tf.where( KL_cond  , KL_loss, tf.zeros_like(KL_loss)) , axis = 0 )

            if bayou_mode:
                self.loss = self.gen_loss + 1/32 * self.loss_RE  + 8/32 * self.gen_loss_FS #+ 256/32 * self.KL_loss
            else:
                self.loss = self.KL_loss +  32 / 256 * (self.gen_loss + 1/32 * self.loss_RE  + 8/32 * self.gen_loss_FS)

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
