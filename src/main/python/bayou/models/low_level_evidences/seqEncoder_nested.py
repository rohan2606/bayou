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

class seqEncoder_nested(object):
    def __init__(self, num_layers, state_size, inputs, batch_size, emb, latent_encoding_variables_intermediate, input_vars_mod_cond):

        with tf.variable_scope('GRU_Encoder_nested'):
            cell_list = []
            for cell in range(num_layers) :
                cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(state_size ) #both default behaviors
                #cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.8,input_keep_prob=0.8,state_keep_prob=0.8)
                cell_list.append(cell)
            cell = tf.contrib.rnn.MultiRNNCell(cell_list)

            # inputs is BS * depth
            inputs = tf.unstack(inputs, axis=1)
            # after unstack it is depth * BS
            input_vars_mod_cond = tf.transpose(input_vars_mod_cond, [1,0]) # depth * BS

            #latent is (modified_batch_size * depth * latent_size)
            latent_encoding_variables_intermediate = tf.transpose(latent_encoding_variables_intermediate, [1,0,2])
            #latent is ( depth * modified_batch_size * latent_size)

            curr_state = [tf.truncated_normal([batch_size, state_size] , stddev=0.001 ) ] * num_layers
            curr_out = tf.zeros([batch_size , state_size])

            for i, inp in enumerate(inputs):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                emb_inp = tf.nn.embedding_lookup(emb, inp) # should be batch * latent_size
                # latent_at_time_i is now batch*latent_size
                emb_inp = tf.concat([emb_inp, latent_encoding_variables_intermediate[i]], axis=1 ) # now this is batch_size * (2*latent_size)

                with tf.variable_scope('cell_complex'):  # handles CHILD_EDGE
                    output, out_state = cell(emb_inp, curr_state)



                cond = tf.logical_and(tf.equal(inp, 0) , tf.equal(input_vars_mod_cond[i], 0) ) #& tf.not_equal(input_vars_mod_cond, 0)
                curr_state = [tf.where(cond, curr_state[j], out_state[j]) for j in range(num_layers)]
                curr_out = tf.where(cond, curr_out, output)

            #
            # with tf.variable_scope("projections"):
            #     projection_w = tf.get_variable('projection_w', [state_size, output_units])
            #     projection_b = tf.get_variable('projection_b', [output_units])

            self.output = curr_out #tf.nn.xw_plus_b(curr_out, projection_w, projection_b)
