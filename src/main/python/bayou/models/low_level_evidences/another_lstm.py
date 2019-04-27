import tensorflow as tf

def while_loop_rnn(rnn_cell1 , rnn_cell2, nodes, edges, initial_state, emb):

    def compute(i, cur_state, out):
        emb_inp = tf.nn.embedding_lookup(emb, nodes[i])

        output1, cur_state1 = rnn_cell1( emb_inp  , cur_state)
        output2, cur_state2 = rnn_cell2( emb_inp  , cur_state)

        cur_state = tf.where(edges[i], cur_state1 , cur_state2)
        output = tf.where(edges[i], output1, output2)
        return i+1, cur_state, out.write(i, output)


    time = tf.shape(nodes)[0]


    _, cur_state, out = tf.while_loop(
        lambda a, b, c: a < time,
        compute,
        (0, initial_state, tf.TensorArray(tf.float32, time)), parallel_iterations=1)
    

    print(out.stack())


    return tf.transpose(out.stack(), [1, 0, 2]), cur_state
    # return out.stack(), cur_state
