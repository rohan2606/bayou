
class TreeEncoder(object):
    def __init__(self, emb, batch_size, num_layers, units, depth, output_units):
        cells1 = []
        cells2 = []
        for _ in range(num_layers):
            cells1.append(tf.nn.rnn_cell.GRUCell(units))
            cells2.append(tf.nn.rnn_cell.GRUCell(units))

        self.cell1 = tf.nn.rnn_cell.MultiRNNCell(cells1)
        self.cell2 = tf.nn.rnn_cell.MultiRNNCell(cells2)

        # placeholders
        # initial_state has get_shape (batch_size, latent_size), same as psi_mean in the prev code
        self.initial_state = [tf.truncated_normal([batch_size, units] , stddev=0.001 ) ] * num_layers
        self.nodes = [tf.placeholder(tf.int32, [batch_size], name='node{0}'.format(i)) for i in range(depth)]
        self.edges = [tf.placeholder(tf.bool, [batch_size], name='edge{0}'.format(i)) for i in range(depth)]

        # projection matrices for output
        with tf.name_scope("projections"):
            self.projection_w = tf.Variable('projection_w', [self.cell1.output_size, output_units])
            self.projection_b = tf.Variable('projection_b', [output_units])

            tf.summary.histogram("projection_w", self.projection_w)
            tf.summary.histogram("projection_b", self.projection_b)


        emb_inp = (tf.nn.embedding_lookup(emb, i) for i in self.nodes)
        self.emb_inp = emb_inp

        # setup embedding
        # setting this variable scope to decoder helps you use the same embedding as in decoder
        with tf.variable_scope('Tree_network'):

            emb_inp = self.emb_inp
            # the decoder (modified from tensorflow's seq2seq library to fit tree RNNs)
            # TODO: update with dynamic decoder (being implemented in tf) once it is released
            with tf.variable_scope('rnn'):
                self.state = self.initial_state
                for i, inp in enumerate(emb_inp):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    with tf.variable_scope('cell1'):  # handles CHILD_EDGE
                        output1, state1 = self.cell1(inp, self.state)
                    with tf.variable_scope('cell2'): # handles SIBLING EDGE
                        output2, state2 = self.cell2(inp, self.state)

                    output = tf.where(self.edges[i], output1, output2)
                    self.state = [tf.where(self.edges[i], state1[j], state2[j]) for j in range(num_layers)]


        with tf.name_scope("Output"):
            self.last_op = tf.nn.xw_plus_b(output, self.projection_zw, self.projection_zb)
