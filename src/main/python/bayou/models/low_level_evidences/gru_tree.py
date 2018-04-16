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

class TreeEncoder(object):
    def __init__(self, emb, batch_size, nodes, edges, num_layers, units, depth, output_units):
        cells1 = []
        cells2 = []
        for _ in range(num_layers):
            cells1.append(tf.nn.rnn_cell.GRUCell(units))
            cells2.append(tf.nn.rnn_cell.GRUCell(units))

        self.cell1 = tf.nn.rnn_cell.MultiRNNCell(cells1)
        self.cell2 = tf.nn.rnn_cell.MultiRNNCell(cells2)

        # placeholders
        # initial_state has get_shape (batch_size, latent_size), same as psi_mean in the prev code
        self.initial_state = [tf.truncated_normal([batch_size, units] , stddev=0.001 ) ] * num_layers

        # projection matrices for output
        with tf.name_scope("projections"):
            self.projection_w = tf.get_variable('projection_w', [self.cell1.output_size, output_units])
            self.projection_b = tf.get_variable('projection_b', [output_units])

            tf.summary.histogram("projection_w", self.projection_w)
            tf.summary.histogram("projection_b", self.projection_b)


        emb_inp = (tf.nn.embedding_lookup(emb, i) for i in nodes)
        self.emb_inp = emb_inp

    
        with tf.variable_scope('Tree_network'):

            emb_inp = self.emb_inp
            # the decoder (modified from tensorflow's seq2seq library to fit tree RNNs)
            # TODO: update with dynamic decoder (being implemented in tf) once it is released
            with tf.variable_scope('rnn'):
                self.state = self.initial_state
                for i, inp in enumerate(emb_inp):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    with tf.variable_scope('cell1'):  # handles CHILD_EDGE
                        output1, state1 = self.cell1(inp, self.state)
                    with tf.variable_scope('cell2'): # handles SIBLING EDGE
                        output2, state2 = self.cell2(inp, self.state)

                    output = tf.where(edges[i], output1, output2)
                    self.state = [tf.where(edges[i], state1[j], state2[j]) for j in range(num_layers)]


        with tf.name_scope("Output"):
            self.last_output = tf.nn.xw_plus_b(output, self.projection_w, self.projection_b)
