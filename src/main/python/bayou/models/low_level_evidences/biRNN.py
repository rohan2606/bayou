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

class biRNN(object):
    def __init__(self, num_layers, state_size, inputs, batch_size, emb, output_units):

        with tf.variable_scope('GRU_Encoder'):
            cell_list_fwd, cell_list_back = [],[]
            for i in range(num_layers) :
                    cell_fwd = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(state_size ) #both default behaviors
                    cell_list_fwd.append(cell_fwd)
                    cell_back = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(state_size ) #both default behaviors
                    cell_list_back.append(cell_back)

            multi_cell_fwd = tf.contrib.rnn.MultiRNNCell(cell_list_fwd)
            multi_cell_back = tf.contrib.rnn.MultiRNNCell(cell_list_back)

            # inputs is BS * depth
            inputs_fwd = tf.unstack(inputs, axis=1)
            # after unstack it is depth * BS
            inputs_back = inputs_fwd[::-1]

            curr_state_fwd = [tf.truncated_normal([batch_size, state_size] , stddev=0.001 ) ] * num_layers
            curr_state_back = [tf.truncated_normal([batch_size, state_size] , stddev=0.001 ) ] * num_layers

            curr_out_fwd = tf.zeros([batch_size , state_size])
            curr_out_back = tf.zeros([batch_size , state_size])

            for i, inp in enumerate(zip(inputs_fwd, inputs_back)):
                #if i > 0:
                #    tf.get_variable_scope().reuse_variables()

                inp_fwd, inp_back = inp
                emb_inp_fwd = tf.nn.embedding_lookup(emb, inp_fwd)
                emb_inp_back = tf.nn.embedding_lookup(emb, inp_back)

                with tf.variable_scope("forward", reuse=tf.AUTO_REUSE):
                    output_fwd, out_state_fwd = multi_cell_fwd(emb_inp_fwd, curr_state_fwd)
                with tf.variable_scope("backward", reuse=tf.AUTO_REUSE):
                    output_back, out_state_back = multi_cell_back(emb_inp_back, curr_state_back)

                curr_state_fwd = [tf.where(tf.not_equal(inp_fwd, 0), out_state_fwd[j], curr_state_fwd[j])
                              for j in range(num_layers)]

                curr_state_back = [tf.where(tf.not_equal(inp_back, 0), out_state_back[j], curr_state_back[j])
                              for j in range(num_layers)]

                curr_out_fwd = tf.where(tf.not_equal(inp_fwd, 0), output_fwd, curr_out_fwd)
                curr_out_back = tf.where(tf.not_equal(inp_back, 0), output_back, curr_out_back)


            temp_out = tf.concat([curr_out_fwd, curr_out_back], axis=1)
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                curr_out = tf.layers.dense(temp_out, output_units, activation=tf.nn.tanh)

            #
            # with tf.variable_scope("projections"):
            #     projection_w = tf.get_variable('projection_w', [state_size, output_units])
            #     projection_b = tf.get_variable('projection_b', [output_units])

            self.output = curr_out #tf.nn.xw_plus_b(curr_out, projection_w, projection_b)
