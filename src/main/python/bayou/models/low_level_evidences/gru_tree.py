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
    def __init__(self, emb, config, node_word_indices_placeholder, left_children_placeholder, right_children_placeholder, output_size):


        with tf.variable_scope('Composition'):
            W1 = tf.get_variable('W1',
                                   [3 * config.reverse_encoder.units, config.reverse_encoder.units])
            b1 = tf.get_variable('b1', [1, config.reverse_encoder.units])

        with tf.variable_scope('Projection'):
             U = tf.get_variable('U', [config.reverse_encoder.units, output_size])
             bs = tf.get_variable('bs', [output_size])

        tensor_array_size = tf.constant(config.reverse_encoder.max_ast_depth+1, dtype=tf.int32)
        tensor_array = [tf.TensorArray(tf.float32, size=tensor_array_size, dynamic_size=False, clear_after_read=False, infer_shape=True) for i in range(config.batch_size)]

        for i in range(config.batch_size):
            tensor_array[i] = tensor_array[i].write(0, tf.zeros([1,config.reverse_encoder.units])) # Can make this trainable

        loop_cond = lambda tensor_array, i: tf.less(i, config.reverse_encoder.max_ast_depth)


        def embed_word( word_index):
            return tf.expand_dims(tf.gather(emb, word_index), axis=0) #returns [1,reverse_encoder.units]


        def combineWBothChildren( left_tensor, right_tensor, word_index):
            embNode = embed_word(word_index)
            combine = tf.nn.relu(tf.matmul(tf.concat([left_tensor, right_tensor, embNode], axis=1), W1) + b1)
            return combine


        def combineLeftChild( left_tensor, word_index):
            embNode = embed_word(word_index)
            fakeNode = embed_word(0)
            combine = tf.nn.relu(tf.matmul(tf.concat([left_tensor, fakeNode, embNode], axis=1), W1) + b1)
            return combine


        def combineRightChild( right_tensor, word_index):
            embNode = embed_word(word_index)
            fakeNode = embed_word(0)
            combine = tf.nn.relu(tf.matmul(tf.concat([fakeNode, right_tensor, embNode], axis=1), W1) + b1)
            return combine

        def loop_body(tensor_array, i):

              for j in range(config.batch_size):
                  node_word_index = node_word_indices_placeholder[j,i] #tf.gather(node_word_indices_placeholder, i)
                  left_child = left_children_placeholder[j,i] #tf.gather(left_children_placeholder, i)
                  right_child = right_children_placeholder[j,i] #tf.gather(right_children_placeholder, i)

                  f1 = lambda: combineWBothChildren(tensor_array[j].read(left_child), tensor_array[j].read(right_child),node_word_index)
                  f2 = lambda: combineLeftChild(tensor_array[j].read(left_child), node_word_index)
                  f3 = lambda: combineRightChild(tensor_array[j].read(right_child), node_word_index)
                  f4 = lambda: embed_word(node_word_index)


                  ifOnlyLeftExist  = tf.constant((left_child != 0) and (right_child == 0), dtype=tf.bool)
                  ifOnlyRightExist = tf.constant((left_child == 0) and (right_child != 0), dtype=tf.bool)
                  ifBothExist = tf.constant((left_child != 0) and (right_child != 0), dtype=tf.bool)
                  node_tensor = tf.case({ifBothExist: f1, ifOnlyLeftExist: f2 , ifOnlyRightExist:f3 },
                            default=f4, exclusive=True)

                  #node_tensor = embed_word(node_word_index)

                  tensor_array[j] = tensor_array[j].write(i, node_tensor)


              i = tf.add(i, 1)
              return tensor_array, i

        tensor_array, _ = tf.while_loop(loop_cond, loop_body, [tensor_array, 1], parallel_iterations=1)
        root_logits=[tf.matmul(tensor_array[j].read(tensor_array[j].size() - 1), U) + bs for j in range(config.batch_size)]




        self.output = tf.concat(root_logits, axis=0)
