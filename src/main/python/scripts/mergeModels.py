import tensorflow as tf
from bayou.models.low_level_evidences.data_reader import Reader
from bayou.models.low_level_evidences.utils import read_config, dump_config, get_var_list, static_plot, get_available_gpus
from bayou.models.low_level_evidences.model import Model
import argparse
import json
import os

def reload(clargs):

    config_file = 'savedFocusModel/config.json'
    clargs.continue_from = True

    with open(config_file) as f:
        config = read_config(json.load(f), chars_vocab=True)

    reader = Reader(clargs, config, dataIsThere=True)


    # Placeholders for tf data
    prog_ids_placeholder = tf.placeholder(reader.prog_ids.dtype, reader.prog_ids.shape)
    js_prog_ids_placeholder = tf.placeholder(reader.js_prog_ids.dtype, reader.js_prog_ids.shape)
    nodes_placeholder = tf.placeholder(reader.nodes.dtype, reader.nodes.shape)
    edges_placeholder = tf.placeholder(reader.edges.dtype, reader.edges.shape)
    targets_placeholder = tf.placeholder(reader.targets.dtype, reader.targets.shape)
    evidence_placeholder = [tf.placeholder(input.dtype, input.shape) for input in reader.inputs]
    # reset batches

    feed_dict={fp: f for fp, f in zip(evidence_placeholder, reader.inputs)}
    feed_dict.update({prog_ids_placeholder: reader.prog_ids})
    feed_dict.update({js_prog_ids_placeholder: reader.js_prog_ids})
    feed_dict.update({nodes_placeholder: reader.nodes})
    feed_dict.update({edges_placeholder: reader.edges})
    feed_dict.update({targets_placeholder: reader.targets})

    dataset = tf.data.Dataset.from_tensor_slices((prog_ids_placeholder, js_prog_ids_placeholder, nodes_placeholder, edges_placeholder, targets_placeholder, *evidence_placeholder))
    batched_dataset = dataset.batch(config.batch_size)
    iterator = batched_dataset.make_initializable_iterator()

    model = Model(config , iterator, bayou_mode=False)
    i = 0
    with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as sess:
        writer = tf.summary.FileWriter(clargs.save)
        writer.add_graph(sess.graph)
        tf.global_variables_initializer().run()

        tf.train.write_graph(sess.graph_def, clargs.save, 'model.pbtxt')
        tf.train.write_graph(sess.graph_def, clargs.save, 'model.pb', as_text=False)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)

        # restore model
        if clargs.continue_from is not None:
            bayou_vars = get_var_list()['bayou_vars']
            old_saver = tf.train.Saver(bayou_vars, max_to_keep=None)
            ckpt = tf.train.get_checkpoint_state('savedFocusModelReDone')
            old_saver.restore(sess, ckpt.model_checkpoint_path)

            reverse_encoder_vars = get_var_list()['rev_encoder_vars']
            old_saver = tf.train.Saver(reverse_encoder_vars, max_to_keep=None)
            ckpt = tf.train.get_checkpoint_state('savedFocusModel')
            old_saver.restore(sess, ckpt.model_checkpoint_path)


        checkpoint_dir = os.path.join(clargs.save, 'model{}.ckpt'.format(i+1))
        saver.save(sess, checkpoint_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--python_recursion_limit', type=int, default=10000,
                        help='set recursion limit for the Python interpreter')
    parser.add_argument('--save', type=str, default='focusModelCombined',
                        help='checkpoint model during training here')
    parser.add_argument('--config', type=str, default=None,
                        help='config file (see description above for help)')
    clargs = parser.parse_args()
    reload(clargs)
