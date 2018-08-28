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

class BayesianEncoder(object):
    def __init__(self, config, inputs):

        # self.inputs = [ev.placeholder(config) for ev in config.evidence]
        exists = [ev.exists(i) for ev, i in zip(config.evidence, inputs)]
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
            self.encodings = [ev.encode(i, config) for ev, i in zip(config.evidence, inputs)]
            encodings = [encoding / tf.square(ev.sigma) for ev, encoding in
                         zip(config.evidence, self.encodings)]

            # 2. pick only encodings from valid inputs that exist, otherwise pick zero encoding
            encodings = [tf.where(exist, enc, zeros) for exist, enc in zip(exists, encodings)]

            # 3. tile the encodings according to each evidence type
            encodings = [[enc] * ev.tile for ev, enc in zip(config.evidence, encodings)]
            encodings = tf.stack(list(chain.from_iterable(encodings)))

            # 4. compute the mean of non-zero encodings
            self.psi_mean = tf.reduce_sum(encodings, axis=0) / denom

        # Compute the covariance of Psi
        with tf.variable_scope('covariance'):
            I = tf.ones([config.batch_size, config.latent_size], dtype=tf.float32)
            self.psi_covariance = I / denom


class BayesianDecoder(object):
    def __init__(self, config, emb, initial_state, nodes, edges, infer=False):

        cells1, cells2 = [], []
        for _ in range(config.decoder.num_layers):
            cells1.append(tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(config.decoder.units))
            cells2.append(tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(config.decoder.units))

        self.cell1 = tf.nn.rnn_cell.MultiRNNCell(cells1)
        self.cell2 = tf.nn.rnn_cell.MultiRNNCell(cells2)

        # placeholders
        self.initial_state = [initial_state] * config.decoder.num_layers
        self.nodes = [nodes[i] for i in range(config.decoder.max_ast_depth)]
        self.edges = [edges[i] for i in range(config.decoder.max_ast_depth)]

        # projection matrices for output
        with tf.variable_scope("projections"):
            self.projection_w = tf.get_variable('projection_w', [self.cell1.output_size,
                                                                 config.decoder.vocab_size])
            self.projection_b = tf.get_variable('projection_b', [config.decoder.vocab_size])
            # tf.summary.histogram("projection_w", self.projection_w)
            # tf.summary.histogram("projection_b", self.projection_b)

        # setup embedding
        emb_inp = (tf.nn.embedding_lookup(emb, i) for i in self.nodes)
        self.emb_inp = emb_inp

        with tf.variable_scope('decoder_network'):
            def loop_fn(prev, _):
                prev = tf.nn.xw_plus_b(prev, self.projection_w, self.projection_b)
                prev_symbol = tf.argmax(prev, 1)
                return tf.nn.embedding_lookup(emb, prev_symbol)
            loop_function = loop_fn if infer else None

            emb_inp = self.emb_inp
            # the decoder (modified from tensorflow's seq2seq library to fit tree RNNs)
            # TODO: update with dynamic decoder (being implemented in tf) once it is released
            with tf.variable_scope('rnn'):

                self.state = self.initial_state
                self.outputs = []
                prev = None
                for i, inp in enumerate(emb_inp):
                    if loop_function is not None and prev is not None:
                        with tf.variable_scope('loop_function', reuse=True):
                            inp = loop_function(prev, i)
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    with tf.variable_scope('cell1'):  # handles CHILD_EDGE
                        output1, state1 = self.cell1(inp, self.state)
                    with tf.variable_scope('cell2'):  # handles SIBLING_EDGE
                        output2, state2 = self.cell2(inp, self.state)
                    output = tf.where(self.edges[i], output1, output2)
                    self.state = [tf.where(self.edges[i], state1[j], state2[j])
                                  for j in range(config.decoder.num_layers)]
                    self.outputs.append(output)
                    if loop_function is not None:
                        prev = output


class BayesianReverseEncoder(object):
    def __init__(self, config, emb, tree_nodes, tree_edges):
        tree_nodes = [[ tree_nodes[j][ config.reverse_encoder.max_ast_depth -1 -i ] for i in range(config.reverse_encoder.max_ast_depth)] for j in range(5)]
        tree_edges = [[ tree_edges[j][ config.reverse_encoder.max_ast_depth -1 -i ] for i in range(config.reverse_encoder.max_ast_depth)] for j in range(5)]

        with tf.variable_scope("Covariance"):
            Covariance_Tree = TreeEncoder(emb, config.batch_size, tree_nodes, tree_edges, config.reverse_encoder.num_layers, \
                config.reverse_encoder.units, config.reverse_encoder.max_ast_depth, 1)
            d = tf.square(Covariance_Tree.last_output)
            d = 1. +  d
            denom = tf.tile(d, [1, config.latent_size])
            I = tf.ones([config.batch_size, config.latent_size], dtype=tf.float32)
            self.psi_covariance = I / denom

        with tf.variable_scope("Mean"):
            Mean_Tree = TreeEncoder(emb, config.batch_size, tree_nodes, tree_edges, config.reverse_encoder.num_layers, \
                                config.reverse_encoder.units, config.reverse_encoder.max_ast_depth, config.latent_size)
            self.psi_mean = Mean_Tree.last_output
