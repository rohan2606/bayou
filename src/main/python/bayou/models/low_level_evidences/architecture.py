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
from itertools import chain
from bayou.models.low_level_evidences.gru_tree import TreeEncoder
from bayou.models.low_level_evidences.seqEncoder import seqEncoder
from another_lstm import while_loop_rnn

class BayesianEncoder(object):
    def __init__(self, config, inputs, infer=False):

        # exists  = #ev * batch_size
        exists = [ev.exists(i, config, infer) for ev, i in zip(config.evidence, inputs)]
        zeros = tf.zeros([config.batch_size, config.latent_size], dtype=tf.float32)

        # Compute the denominator used for mean and covariance
        for ev in config.evidence:
            ev.init_sigma(config)

        d = [tf.where(exist, tf.tile([1. / tf.square(ev.sigma)], [config.batch_size]),
                      tf.zeros(config.batch_size)) for ev, exist in zip(config.evidence, exists)]
        d = 1. + tf.reduce_sum(tf.stack(d), axis=0)
        denom = tf.tile(tf.reshape(d, [-1, 1]), [1, config.latent_size])

        # Compute the mean of Psi
        with tf.variable_scope('mean'):
            # 1. compute encoding

            encodings = [ev.encode(i, config, infer) for ev, i in zip(config.evidence, inputs)]
            encodings = [encoding / tf.square(ev.sigma) for ev, encoding in
                         zip(config.evidence, encodings)]

            # 2. pick only encodings from valid inputs that exist, otherwise pick zero encoding
            self.encodings = [tf.where(exist, enc, zeros) for exist, enc in zip(exists, encodings)]

            # 3. tile the encodings according to each evidence type
            encodings = [[enc] * ev.tile for ev, enc in zip(config.evidence, self.encodings)]
            encodings = tf.stack(list(chain.from_iterable(encodings)))

            # 4. compute the mean of non-zero encodings
            self.psi_mean = tf.reduce_sum(encodings, axis=0) / denom

        # Compute the covariance of Psi
        with tf.variable_scope('covariance'):
            I = tf.ones([config.batch_size, config.latent_size], dtype=tf.float32)
            self.psi_covariance = I / denom


class BayesianDecoder(object):
    def __init__(self, config, emb, initial_state, nodes, edges):

        self.cell1 = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(config.decoder.units) #tf.nn.rnn_cell.MultiRNNCell(cells1)
        self.cell2 = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(config.decoder.units)


        self.outputs, state = while_loop_rnn(self.cell1, self.cell2, nodes, edges, initial_state, emb)


        # placeholders
        # self.nodes = [nodes[i] for i in range(config.decoder.max_ast_depth)]
        # self.edges = [edges[i] for i in range(config.decoder.max_ast_depth)]

        # projection matrices for output
        with tf.variable_scope("projections"):
            self.projection_w = tf.get_variable('projection_w', [self.cell1.output_size,
                                                                 config.decoder.vocab_size])
            self.projection_b = tf.get_variable('projection_b', [config.decoder.vocab_size])


        # setup embedding
        #
        # with tf.variable_scope('decoder_network'):
        #     # the decoder (modified from tensorflow's seq2seq library to fit tree RNNs)
        #     with tf.variable_scope('rnn'):
        #
        #         self.state = self.initial_state
        #         self.outputs = []
        #         for i, inp in enumerate(emb_inp):
        #             if i > 0:
        #                 tf.get_variable_scope().reuse_variables()
        #             with tf.variable_scope('cell1'):  # handles CHILD_EDGE
        #                 output1, state1 = self.cell1(inp, self.state)
        #             with tf.variable_scope('cell2'):  # handles SIBLING_EDGE
        #                 output2, state2 = self.cell2(inp, self.state)
        #             output = tf.where(self.edges[i], output1, output2)
        #             self.state = [tf.where(self.edges[i], state1[j], state2[j])
        #                           for j in range(config.decoder.num_layers)]
        #             self.outputs.append(output)


class SimpleDecoder(object):
    def __init__(self, config, emb, initial_state, nodes, ev_config):

        cells1 = []
        for _ in range(config.decoder.num_layers):
            cells1.append(tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(ev_config.units))

        self.cell1 = tf.nn.rnn_cell.MultiRNNCell(cells1)

        # placeholders
        self.initial_state = [initial_state] * ev_config.num_layers
        self.nodes = [nodes[i] for i in range(ev_config.max_depth)]

        # projection matrices for output
        with tf.variable_scope("projections_FS"):
            self.projection_w_FS = tf.get_variable('projection_w_FS', [self.cell1.output_size,
                                                                 ev_config.vocab_size])
            self.projection_b_FS = tf.get_variable('projection_b_FS', [ev_config.vocab_size])
            # tf.summary.histogram("projection_w", self.projection_w)
            # tf.summary.histogram("projection_b", self.projection_b)

        # setup embedding
        emb_inp = (tf.nn.embedding_lookup(emb, i) for i in self.nodes)

        with tf.variable_scope('decoder_network_FS'):
            # the decoder (modified from tensorflow's seq2seq library to fit tree RNNs)
            with tf.variable_scope('rnn_FS'):

                self.state = self.initial_state
                self.outputs = []
                for i, inp in enumerate(emb_inp):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    with tf.variable_scope('cell1_FS'):  # handles CHILD_EDGE
                        output1, state1 = self.cell1(inp, self.state)
                    output =  output1
                    self.state = [  state1[j] for j in range(ev_config.num_layers)]
                    self.outputs.append(output)


class BayesianReverseEncoder(object):
    def __init__(self, config, emb, nodes, edges, returnType, embRE, formalParam, embFP):

        nodes = [ nodes[i] for i in range(config.reverse_encoder.max_ast_depth)]
        edges = [ edges[i] for i in range(config.reverse_encoder.max_ast_depth)]

        with tf.variable_scope("Covariance"):
            with tf.variable_scope("APITree"):
                API_Cov_Tree = TreeEncoder(emb, config.batch_size, nodes, edges, config.reverse_encoder.num_layers, \
                                    config.reverse_encoder.units, config.reverse_encoder.max_ast_depth, config.latent_size)
                Tree_Cov = API_Cov_Tree.last_output

            with tf.variable_scope('ReturnType'):
                Ret_Seq = seqEncoder(config.reverse_encoder.num_layers, config.reverse_encoder.units, returnType, config.batch_size, embRE, 1)

                w = tf.get_variable('w', [config.reverse_encoder.units, config.latent_size ])
                b = tf.get_variable('b', [config.latent_size])

                rt_Cov = tf.nn.xw_plus_b(Ret_Seq.output ,w, b)


            with tf.variable_scope('FormalParam'):
                fp_Seq = seqEncoder(config.reverse_encoder.num_layers, config.reverse_encoder.units, formalParam, config.batch_size, embFP, 1)

                w = tf.get_variable('w', [config.reverse_encoder.units, config.latent_size ])
                b = tf.get_variable('b', [config.latent_size])

                fp_Cov = tf.nn.xw_plus_b(fp_Seq.output,w, b)


        with tf.variable_scope("Mean"):
            with tf.variable_scope('APITree'):
                API_Mean_Tree = TreeEncoder(emb, config.batch_size, nodes, edges, config.reverse_encoder.num_layers, \
                                    config.reverse_encoder.units, config.reverse_encoder.max_ast_depth, config.latent_size)
                Tree_mean = API_Mean_Tree.last_output

            with tf.variable_scope('ReturnType'):
                Ret_Seq = seqEncoder(config.reverse_encoder.num_layers, config.reverse_encoder.units, returnType, config.batch_size, embRE, config.latent_size)

                w = tf.get_variable('w', [config.reverse_encoder.units, config.latent_size])
                b = tf.get_variable('b', [config.latent_size])

                rt_mean = tf.nn.xw_plus_b(Ret_Seq.output ,w, b)


            with tf.variable_scope('FormalParam'):
                fp_Seq = seqEncoder(config.reverse_encoder.num_layers, config.reverse_encoder.units, formalParam, config.batch_size, embFP, config.latent_size)

                w = tf.get_variable('w', [config.reverse_encoder.units, config.latent_size])
                b = tf.get_variable('b', [config.latent_size])

                fp_mean = tf.nn.xw_plus_b(fp_Seq.output,w, b)


            sigmas = [Tree_Cov , rt_Cov, fp_Cov]

            #dimension is  3*batch * 1
            finalSigma = tf.layers.dense(tf.reshape( tf.transpose(tf.stack(sigmas, axis=0), perm=[1,0,2]), [config.batch_size, -1]) , config.latent_size, activation=tf.nn.tanh)
            finalSigma = tf.layers.dense(finalSigma, config.latent_size, activation=tf.nn.tanh)

            finalSigma = tf.layers.dense(finalSigma, 1)

            d = tf.tile(tf.square(finalSigma),[1, config.latent_size])
            d = .00000001 + d
            #denom = d # tf.tile(tf.reshape(d, [-1, 1]), [1, config.latent_size])
            #I = tf.ones([config.batch_size, config.latent_size], dtype=tf.float32)
            self.psi_covariance = d #I / denom

            encodings = [Tree_mean, rt_mean, fp_mean]
            finalMean = tf.layers.dense(tf.reshape( tf.transpose(tf.stack(encodings, axis=0), perm=[1,0,2]), [config.batch_size, -1]) , config.latent_size, activation=tf.nn.tanh)
            finalMean = tf.layers.dense(finalMean, config.latent_size, activation=tf.nn.tanh)
            finalMean = tf.layers.dense(finalMean, config.latent_size)
            # 4. compute the mean of non-zero encodings
            self.psi_mean = finalMean
