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
            cells1.append(tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(units))
            cells2.append(tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(units))

        self.cell1 = tf.nn.rnn_cell.MultiRNNCell(cells1)
        self.cell2 = tf.nn.rnn_cell.MultiRNNCell(cells2)

        # initial_state has get_shape (batch_size, latent_size), same as psi_mean in the prev code
        curr_state = [tf.truncated_normal([batch_size, units] , stddev=0.001 ) ] * num_layers
        curr_out = tf.zeros([batch_size , units])

        # projection matrices for output
        with tf.name_scope("projections"):
            self.projection_w = tf.get_variable('projection_w', [self.cell1.output_size, output_units])
            self.projection_b = tf.get_variable('projection_b', [output_units])


        emb_inp = (tf.nn.embedding_lookup(emb, i) for i in nodes)

        with tf.variable_scope('Tree_network'):

            # the decoder (modified from tensorflow's seq2seq library to fit tree RNNs)
            with tf.variable_scope('rnn'):
                self.state = curr_state
                for i, inp in enumerate(emb_inp):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    with tf.variable_scope('cell1'):  # handles CHILD_EDGE
                        output1, state1 = self.cell1(inp, self.state)
                    with tf.variable_scope('cell2'): # handles SIBLING EDGE
                        output2, state2 = self.cell2(inp, self.state)

                    output = tf.where(edges[i], output1, output2)
                    curr_out = tf.where(tf.not_equal(inp, 0), output, curr_out)

                    self.state = [tf.where(edges[i], state1[j], state2[j]) for j in range(num_layers)]
                    curr_state = [tf.where(tf.not_equal(inp, 0), self.state[j], curr_state[j])
                              for j in range(num_layers)]

        with tf.name_scope("Output"):
            self.last_output = tf.nn.xw_plus_b(curr_out, self.projection_w, self.projection_b)
