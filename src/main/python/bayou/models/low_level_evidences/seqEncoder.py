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
    def __init__(self, num_layers, state_size, inputs):
        with tf.variable_scope('LSTM_Encoder'):
            cell_list = []
            for cell in range(num_layers) :
                cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell (state_size ) #both default behaviors
                #cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.8,input_keep_prob=0.8,state_keep_prob=0.8)
                cell_list.append(cell)
            cell = tf.contrib.rnn.MultiRNNCell(cell_list)
            inputs = tf.unstack(inputs, axis=1)
            outputs, current_state = tf.nn.static_rnn(cell, inputs, dtype=tf.float32)
            output = outputs[-1]
            self.output = output
