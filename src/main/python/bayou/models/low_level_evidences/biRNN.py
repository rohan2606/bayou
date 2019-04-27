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

            cell_fwd = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(state_size )
            cell_back = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(state_size )

            # inputs is BS * depth
            inputs = tf.transpose(inputs)
            inputs_back = tf.transpose(tf.reverse_v2(inputs, [1]))

            init_state_fwd = tf.truncated_normal([batch_size, state_size] , stddev=0.001 )
            init_state_back = tf.truncated_normal([batch_size, state_size] , stddev=0.001 )



            def compute_fwd(i, cur_state, out):
                emb_inp = tf.nn.embedding_lookup(emb, inputs[i])
                output, cur_state = cell_fwd( emb_inp  , cur_state)
                return i+1, cur_state, out.write(i, output)



            def compute_back(i, cur_state, out):
                emb_inp = tf.nn.embedding_lookup(emb, inputs[i])
                output, cur_state = cell_back( emb_inp  , cur_state)
                return i+1, cur_state, out.write(i, output)


            time = tf.shape(inputs)[0]


            _, cur_state_fwd, curr_out_fwd = tf.while_loop(
                lambda a, b, c: a < time,
                compute_fwd,
                (0, init_state_fwd, tf.TensorArray(tf.float32, time)), parallel_iterations=1)


            _, cur_state_back, curr_out_back = tf.while_loop(
                lambda a, b, c: a < time,
                compute_back,
                (0, init_state_back, tf.TensorArray(tf.float32, time)), parallel_iterations=1)




            temp_out = tf.concat([curr_out_fwd.stack()[-1], curr_out_back.stack()[-1]], axis=1)
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                curr_out = tf.layers.dense(temp_out, output_units, activation=tf.nn.tanh)


            with tf.variable_scope("projections"):
                projection_w = tf.get_variable('projection_w', [state_size, output_units])
                projection_b = tf.get_variable('projection_b', [output_units])


            self.output = tf.nn.xw_plus_b(curr_out, projection_w, projection_b)
