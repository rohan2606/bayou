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
	def __init__(self, emb, batch_size, tree_nodes, tree_edges, num_layers, units, depth, output_units):
		#tree nodes / edges are (batch_size * 5) * depth
		exists = tf.not_equal(tf.reduce_sum(tree_nodes, axis=0) , 0)

		cells1 = []
		cells2 = []
		for _ in range(num_layers):
			cells1.append(tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(units))
			cells2.append(tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(units))

		self.cell1 = tf.nn.rnn_cell.MultiRNNCell(cells1)
		self.cell2 = tf.nn.rnn_cell.MultiRNNCell(cells2)

		curr_state = [tf.truncated_normal([batch_size*5, units] , stddev=0.001 ) ] * num_layers
		curr_out = tf.zeros([batch_size*5 , units])

		# initial_state has get_shape (batch_size * 5, latent_size), same as psi_mean in the prev code

        # projection matrices for output
		with tf.name_scope("projections"):
			self.merger_w = tf.get_variable('merger_w', [5*units, units])
			self.merger_b = tf.get_variable('merger_b', [units])

			self.projection_w = tf.get_variable('projection_w', [units, output_units])
			self.projection_b = tf.get_variable('projection_b', [output_units])

		self.last_outputs = []
		# self.states = []
		emb_inp = tf.unstack(tree_nodes, axis=0) #tf.map_fn(lambda i: tf.nn.embedding_lookup(emb, i) , tree_nodes)
		#emb_inp = (tf.nn.embedding_lookup(emb, i) for i in emb_inp)
		# the decoder (modified from tensorflow's seq2seq library to fit tree RNNs)
		# TODO: update with dynamic decoder (being implemented in tf) once it is released
		with tf.variable_scope('gru_tree'):
			self.state = curr_state
			for i, inp in enumerate(emb_inp):
				
				inp_emb = tf.nn.embedding_lookup(emb, inp)
				if i > 0:
					tf.get_variable_scope().reuse_variables()
				with tf.variable_scope('cell1'):  # handles CHILD_EDGE
					output1, state1 = self.cell1(inp_emb, self.state)
				with tf.variable_scope('cell2'): # handles SIBLING EDGE
					output2, state2 = self.cell2(inp_emb, self.state)
				
				self.output = tf.where(tree_edges[i], output1, output2)
				curr_out = tf.where(tf.not_equal(inp, 0), self.output, curr_out)

				self.state = [tf.where(tree_edges[i], state1[j], state2[j]) for j in range(num_layers)]
				curr_state = [tf.where(tf.not_equal(inp, 0), self.state[j], curr_state[j]) for j in range(num_layers)]



		# where keeps it (bs * 5)* units
		zeros = tf.zeros([batch_size * 5, units])
		self.output = tf.where(exists, curr_out , zeros)

		temp = tf.reshape(self.output , [batch_size, 5 * units ])
		#temp = tf.reduce_mean(temp, axis=1)
		merged_last_op = tf.nn.tanh(tf.nn.xw_plus_b(temp ,  self.merger_w, self.merger_b))
		merged_last_op = tf.layers.dense(merged_last_op, units, activation=tf.nn.tanh) 
		merged_last_op = tf.layers.dense(merged_last_op, units, activation=tf.nn.tanh) 
		#merged_last_op is batch_size * units

		self.last_output = tf.nn.xw_plus_b(merged_last_op, self.projection_w, self.projection_b)
		return
