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
    def __init__(self, config, nodes, edges, output_units):


        emb = tf.get_variable('RE_api_call_emb', [config.reverse_encoder.vocab_size, config.reverse_encoder.units])
        cell1 = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(config.reverse_encoder.units) #tf.nn.rnn_cell.MultiRNNCell(cells1)
        cell2 = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(config.reverse_encoder.units)

        initial_state = tf.truncated_normal([config.batch_size, config.reverse_encoder.units] , stddev=0.001 )

        def compute(i, cur_state, out):
            emb_inp = tf.nn.embedding_lookup(emb, nodes[i])

            output1, cur_state1 = cell1( emb_inp  , cur_state)
            output2, cur_state2 = cell2( emb_inp  , cur_state)

            cur_state = tf.where(edges[i], cur_state1 , cur_state2)
            output = tf.where(edges[i], output1, output2)

            return i+1, cur_state, out.write(i, output)



        time = tf.shape(nodes)[0]

        _, cur_state, out = tf.while_loop(
            lambda a, b, c: a < time,
            compute,
            (0, initial_state, tf.TensorArray(tf.float32, time)), parallel_iterations=1)


        with tf.name_scope("Output"):
            projection_w = tf.get_variable('projection_w', [cell1.output_size, output_units])
            projection_b = tf.get_variable('projection_b', [output_units])

            self.last_output = tf.nn.xw_plus_b(out.stack()[-1], projection_w, projection_b)

        return
