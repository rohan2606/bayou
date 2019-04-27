import tensorflow as tf

class TreeDecoder():

    def __init__(self, config, nodes, edges, initial_state, emb):
        cell1 = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(config.decoder.units) #tf.nn.rnn_cell.MultiRNNCell(cells1)
        cell2 = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(config.decoder.units)


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


        self.all_outputs  =  tf.transpose(out.stack(), [1, 0, 2])

        return
