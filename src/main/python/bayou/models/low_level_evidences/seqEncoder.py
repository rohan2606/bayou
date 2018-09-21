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

            curr_state = [tf.truncated_normal([batch_size, state_size] , stddev=0.001 ) ] * num_layers
            curr_out = tf.zeros([batch_size , state_size])

            for i, inp in enumerate(inputs):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                emb_inp = tf.nn.embedding_lookup(emb, inp)

                with tf.variable_scope('cell0'):  # handles CHILD_EDGE
                    output, out_state = cell(emb_inp, curr_state)

                curr_state = [tf.where(tf.not_equal(inp, 0), out_state[j], curr_state[j])
                              for j in range(num_layers)]
                curr_out = tf.where(tf.not_equal(inp, 0), output, curr_out)


            self.output = curr_out
