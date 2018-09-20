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

class seqEncoder(object):
    def __init__(self, num_layers, state_size, inputs, batch_size, emb):
        with tf.variable_scope('GRU_Encoder'):
            cell_list = []
            for cell in range(num_layers) :
                cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(state_size ) #both default behaviors
                #cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.8,input_keep_prob=0.8,state_keep_prob=0.8)
                cell_list.append(cell)
            cell = tf.contrib.rnn.MultiRNNCell(cell_list)

            # inputs is BS * depth
            inputs = tf.unstack(inputs, axis=1)
            # after unstack it is depth * BS

            initial_state = [tf.truncated_normal([batch_size, state_size] , stddev=0.001 ) ] * num_layers

            state = initial_state
            outputs = []

            default_out = tf.zeros([batch_size , state_size])
            for i, inp in enumerate(inputs):
                emb_inp = tf.nn.embedding_lookup(emb, inp)
                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                with tf.variable_scope('cell0'):  # handles CHILD_EDGE
                    output, state1 = cell(emb_inp, state)

                output = tf.where(tf.not_equal(inp, 0), output, default_out)
                state = [tf.where(tf.not_equal(inp, 0), state1[j], initial_state[j])
                              for j in range(num_layers)]
                outputs.append(output)


            output = outputs[-1]
            self.output = output
