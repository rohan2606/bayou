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
		#tree nodes / edges are 5* depth* batch_size
		# reshape makes it bs * 5
		exists = tf.reshape(tf.transpose(tf.not_equal(tf.reduce_sum(tree_nodes, axis=1) , 0), perm=[1,0]), [-1,5])
		# tile makes it bs*5* units
		exists = tf.tile(tf.expand_dims(exists, axis=2) , [1,1,units])

		cells1 = []
		cells2 = []
		for _ in range(num_layers):
			cells1.append(tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(units))
			cells2.append(tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(units))

		self.cell1 = tf.nn.rnn_cell.MultiRNNCell(cells1)
		self.cell2 = tf.nn.rnn_cell.MultiRNNCell(cells2)

		# initial_state has get_shape (batch_size, latent_size), same as psi_mean in the prev code
		self.initial_state = [[tf.truncated_normal([batch_size, units] , stddev=0.001 ) ] * num_layers for j in range(5)]

        # projection matrices for output
		with tf.name_scope("projections"):
			self.merger_w = tf.get_variable('merger_w', [5*units, units])
			self.merger_b = tf.get_variable('merger_b', [units])

			self.projection_w = tf.get_variable('projection_w', [units, output_units])
			self.projection_b = tf.get_variable('projection_b', [output_units])

		self.last_outputs = []
		# self.states = []
		for j, nodes, edges in zip(range(5), tree_nodes, tree_edges):
			if j > 0:
				tf.get_variable_scope().reuse_variables()
			emb_inp = (tf.nn.embedding_lookup(emb, i) for i in nodes)

			# the decoder (modified from tensorflow's seq2seq library to fit tree RNNs)
			# TODO: update with dynamic decoder (being implemented in tf) once it is released
			with tf.variable_scope('gru_tree'):
				self.state = self.initial_state[j]
				for i, inp in enumerate(emb_inp):
					if i > 0:
						tf.get_variable_scope().reuse_variables()
					with tf.variable_scope('cell1'):  # handles CHILD_EDGE
						output1, state1 = self.cell1(inp, self.state)
					with tf.variable_scope('cell2'): # handles SIBLING EDGE
						output2, state2 = self.cell2(inp, self.state)

						output = tf.where(edges[i], output1, output2)
						self.state = [tf.where(edges[i], state1[j], state2[j]) for j in range(num_layers)]
			# self.states.append(self.state)

			self.last_outputs.append(output)

		#stack makes it 5*batch_size*units
		# transpose makes it batch * 5 * units
		temp = tf.transpose(tf.stack(self.last_outputs), perm=[1,0,2])

		# where keeps it bs * 5* units
		zeros = tf.zeros([batch_size,5,units])
		temp = tf.where(exists, temp, zeros)

		temp = tf.reshape(temp , [-1, units * 5 ])
		merged_last_op = tf.nn.tanh(tf.nn.xw_plus_b(temp ,  self.merger_w, self.merger_b))
		#merged_last_op is batch_size * units


		self.last_output = tf.nn.xw_plus_b(merged_last_op, self.projection_w, self.projection_b)
		return
