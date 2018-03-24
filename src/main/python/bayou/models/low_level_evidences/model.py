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
from bayou.models.low_level_evidences.data_reader import CHILD_EDGE, SIBLING_EDGE


class Model():
    def __init__(self, config, infer=False):
        assert config.model == 'lle', 'Trying to load different model implementation: ' + config.model
        self.config = config
        if infer:
            config.batch_size = 1
            # THE NEXT LINE IS TO BE TREATED WITH CAUTION FOR CODE SEARCH
            #config.decoder.max_ast_depth = 1


        #setup the encode, however remember that the Encoder is there only for KL-divergence
        # Also note that no samples were generated from it
        self.encoder = BayesianEncoder(config)
        # Note that psi_encoder and samples2 are only used in inference
        samples_2 = tf.random_normal([config.batch_size, config.latent_size],
                                   mean=0., stddev=1., dtype=tf.float32)
        self.psi_encoder = self.encoder.psi_mean + tf.sqrt(self.encoder.psi_covariance) * samples_2


        # setup the reverse encoder.
        self.reverse_encoder = BayesianReverseEncoder(config, self.encoder.psi_covariance)
        samples = tf.random_normal([config.batch_size, config.latent_size],
                                   mean=0., stddev=1., dtype=tf.float32)
        self.psi_reverse_encoder = self.reverse_encoder.psi_mean + tf.sqrt(self.reverse_encoder.psi_covariance) * samples

        # setup the decoder with psi as the initial state
        lift_w = tf.get_variable('lift_w', [config.latent_size, config.decoder.units])
        lift_b = tf.get_variable('lift_b', [config.decoder.units])
        self.initial_state = tf.nn.xw_plus_b(self.psi_reverse_encoder, lift_w, lift_b)

        self.decoder = BayesianDecoder(config, initial_state=self.initial_state, infer=infer)

        # get the decoder outputs
        output = tf.reshape(tf.concat(self.decoder.outputs, 1),
                            [-1, self.decoder.cell1.output_size])
        logits = tf.matmul(output, self.decoder.projection_w) + self.decoder.projection_b
        self.ln_probs = tf.nn.log_softmax(logits)

        # 1. generation loss: log P(X | \Psi)
        self.targets = tf.placeholder(tf.int32, [config.batch_size, config.decoder.max_ast_depth])
        self.gen_loss = seq2seq.sequence_loss([logits], [tf.reshape(self.targets, [-1])],
                                              [tf.ones([config.batch_size * config.decoder.max_ast_depth])])

        # 2. latent loss: negative of the KL-divergence between P(\Psi | f(\Theta)) and P(\Psi)
        #remember, we are minimizing the loss, but derivations were to maximize the lower bound and hence no negative sign
        # KL loss
        KL_loss = 0.5 * tf.reduce_sum( tf.log(self.encoder.psi_covariance) - tf.log(self.reverse_encoder.psi_covariance)
                                          - 1 + self.reverse_encoder.psi_covariance / self.encoder.psi_covariance
                                          + tf.square(self.encoder.psi_mean - self.reverse_encoder.psi_mean)/self.encoder.psi_covariance
                                          , axis=1)
        self.KL_loss = KL_loss


        # The optimizer
        self.loss = self.gen_loss + self.KL_loss
        self.train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.loss)

        var_params = [np.prod([dim.value for dim in var.get_shape()])
                      for var in tf.trainable_variables()]
        if not infer:
            print('Model parameters: {}'.format(np.sum(var_params)))

    # called from infer only when the model is used in inference
    def infer_psi_encoder(self, sess, evidences):
        # Qs: What is evidences. I believe a JSON with headings as keywords, apicalls etc
        # read and wrangle (with batch_size 1) the data
        ## NO NEED TO WRANGLE THE INPUTS ANYMORE
        # inputs = [ev.wrangle(evidences) for ev in self.config.evidence]

        inputs = evidences
        # setup initial states and feed
        feed = {}
        for j, ev in enumerate(self.config.evidence):
            feed[self.encoder.inputs[j].name] = inputs[j]

        psi_encoder, psi_encoder_mean, psi_encoder_Sigma = \
                    sess.run([self.psi_encoder, self.encoder.psi_mean, self.encoder.psi_covariance], feed)

        #sigma is diag of covar, ie std.dev ^ 2
        return psi_encoder, psi_encoder_mean, psi_encoder_Sigma


    def infer_psi_reverse_encoder(self, sess, nodes, edges, evidences):

        # setup initial states and feed
        inputs = evidences
        feed = {}
        for j, ev in enumerate(self.config.evidence):
            feed[self.encoder.inputs[j].name] = inputs[j]
        for j in range(self.config.decoder.max_ast_depth):
            feed[self.reverse_encoder.nodes[j].name] = nodes[j]
            feed[self.reverse_encoder.edges[j].name] = edges[j]

        psi_reverse_encoder, psi_reverse_encoder_mean, psi_reverse_encoder_sigma_sqr = \
                    sess.run([self.psi_reverse_encoder, self.reverse_encoder.psi_mean, self.reverse_encoder.psi_covariance], feed)
        psi_reverse_encoder_sigma = np.sqrt(psi_reverse_encoder_sigma_sqr)

        return psi_reverse_encoder, psi_reverse_encoder_mean, psi_reverse_encoder_sigma

    def infer_lnprobY_given_psi(self, sess, psi, nodes, edges, targets):

        state = sess.run(self.initial_state, {self.psi_reverse_encoder: psi})
        state = [state] * self.config.decoder.num_layers
        prob = 0
        feed = {}
        for j in range(self.config.decoder.max_ast_depth):
            feed[self.decoder.nodes[j].name] = nodes[j]
            feed[self.decoder.edges[j].name] = edges[j]

        for i in range(self.config.decoder.num_layers):
            feed[self.decoder.initial_state[i].name] = state[i]

        [ln_probs, state] = sess.run([self.ln_probs, self.decoder.state], feed)

        for j in range(self.config.decoder.max_ast_depth):
            prob += ln_probs[j][targets[0][j]]

        return prob # this is assumed to be for batch_size = 1
