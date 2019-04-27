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
    def __init__(self, units, inputs, batch_size, output_units, emb):


            inputs = tf.transpose(inputs)
            cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(units ) #both default behaviors

            initial_state = tf.truncated_normal([batch_size, units] , stddev=0.001 )
            curr_out = tf.zeros([batch_size , units])



            def compute(i, cur_state, out):
                emb_inp = tf.nn.embedding_lookup(emb, inputs[i])
                output, cur_state = cell( emb_inp  , cur_state)
                return i+1, cur_state, out.write(i, output)


            time = tf.shape(inputs)[0]


            _, cur_state, out = tf.while_loop(
                lambda a, b, c: a < time,
                compute,
                (0, initial_state, tf.TensorArray(tf.float32, time)), parallel_iterations=1)




            with tf.variable_scope("projections"):
                projection_w = tf.get_variable('projection_w', [units, output_units])
                projection_b = tf.get_variable('projection_b', [output_units])




            #self.last_output = tf.nn.xw_plus_b(out.stack()[-1], projection_w, projection_b)
            self.output = tf.nn.xw_plus_b( out.stack()[-1] , projection_w, projection_b)
