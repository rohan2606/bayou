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

from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
import sys
import json
import textwrap

from bayou.models.low_level_evidences.data_reader import Reader
from bayou.models.low_level_evidences.model import Model
from bayou.models.low_level_evidences.utils import read_config, dump_config
from tensorflow.contrib.tensorboard.plugins import projector

HELP = """\
Config options should be given as a JSON file (see config.json for example):
{                                         |
    "model": "lle"                        | The implementation id of this model (do not change)
    "latent_size": 32,                    | Latent dimensionality
    "batch_size": 50,                     | Minibatch size
    "num_epochs": 100,                    | Number of training epochs
    "learning_rate": 0.02,                | Learning rate
    "print_step": 1,                      | Print training output every given steps
    "alpha": 1e-05,                       | Hyper-param associated with KL-divergence loss
    "beta": 1e-05,                        | Hyper-param associated with evidence loss
    "evidence": [                         | Provide each evidence type in this list
        {                                 |
            "name": "apicalls",           | Name of evidence ("apicalls")
            "units": 64,                  | Size of the encoder hidden state
            "num_layers": 3               | Number of densely connected layers
            "tile": 1                     | Repeat the encoding n times (to boost its signal)
        },                                |
        {                                 |
            "name": "types",              | Name of evidence ("types")
            "units": 32,                  | Size of the encoder hidden state
            "num_layers": 3               | Number of densely connected layers
            "tile": 1                     | Repeat the encoding n times (to boost its signal)
        },                                |
        {                                 |
            "name": "keywords",           | Name of evidence ("keywords")
            "units": 64,                  | Size of the encoder hidden state
            "num_layers": 3               | Number of densely connected layers
            "tile": 1                     | Repeat the encoding n times (to boost its signal)
        }                                 |
    ],                                    |
    "decoder": {                          | Provide parameters for the decoder here
        "units": 256,                     | Size of the decoder hidden state
        "num_layers": 3,                  | Number of layers in the decoder
        "max_ast_depth": 32               | Maximum depth of the AST (length of the longest path)
    }
    "reverse_encoder": {
        "units": 256,
        "num_layers": 3,
        "max_ast_depth": 32
    }                                   |
}                                         |
"""
#%%

PATH = os.getcwd()

LOG_DIR = PATH + '/save'

def train(clargs):
    config_file = clargs.config if clargs.continue_from is None \
                                else os.path.join(clargs.continue_from, 'config.json')
    with open(config_file) as f:
        config = read_config(json.load(f), chars_vocab=clargs.continue_from)
    reader = Reader(clargs, config)

    jsconfig = dump_config(config)
    # print(clargs)
    # print(json.dumps(jsconfig, indent=2))

    with open(os.path.join(clargs.save, 'config.json'), 'w') as f:
        json.dump(jsconfig, fp=f, indent=2)

    model = Model(config)
    merged_summary = tf.summary.merge_all()


    with tf.Session() as sess:
        writer = tf.summary.FileWriter(LOG_DIR)
        writer.add_graph(sess.graph)
        tf.global_variables_initializer().run()

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
        tf.train.write_graph(sess.graph_def, clargs.save, 'model.pbtxt')
        tf.train.write_graph(sess.graph_def, clargs.save, 'model.pb', as_text=False)

        # restore model
        if clargs.continue_from is not None:
            ckpt = tf.train.get_checkpoint_state(clargs.continue_from)
            saver.restore(sess, ckpt.model_checkpoint_path)

        # training
        for i in range(config.num_epochs):
            reader.reset_batches()
            avg_loss = 0
            avg_gen_loss = 0


            for b in range(config.num_batches):
                start = time.time()

                # setup the feed dict
                ev_data, n, e, y = reader.next_batch()
                feed = {model.targets: y}
                for j, ev in enumerate(config.evidence):
                    feed[model.encoder.inputs[j].name] = ev_data[j]
                for j in range(config.decoder.max_ast_depth):
                    feed[model.decoder.nodes[j].name] = n[j]
                    feed[model.decoder.edges[j].name] = e[j]
                    # Feeding value into reverse encoder
                    feed[model.reverse_encoder.nodes[j].name] = n[j]
                    feed[model.reverse_encoder.edges[j].name] = e[j]

                # run the optimizer
                loss, gen_loss, mean, other_mean, _ \
                    = sess.run([model.loss,
                                model.gen_loss,
                                model.encoder.psi_mean,
                                model.reverse_encoder.psi_mean,
                                model.train_op], feed)


                s = sess.run(merged_summary, feed)
                writer.add_summary(s,i)

                end = time.time()
                avg_loss += np.mean(loss)
                avg_gen_loss += np.mean(gen_loss)
                step = i * config.num_batches + b
                if step % config.print_step == 0:
                    print('{}/{} (epoch {}) '
                          'loss: {:.3f}, gen_loss: {:.3f}, mean: {:.3f}, other_mean: {:.3f}, time: {:.3f}'.format
                          (step, config.num_epochs * config.num_batches, i,
                           np.mean(loss),
                           np.mean(gen_loss),
                           np.mean(mean),
                           np.mean(other_mean),
                           end - start))

            if (i+1) % config.checkpoint_step == 0 and i > 0:
                checkpoint_dir = os.path.join(clargs.save, 'model{}.ckpt'.format(i+1))
                saver.save(sess, checkpoint_dir)
                print('Model checkpointed: {}. Average for epoch , '
                      'loss: {:.3f}'.format
                      (checkpoint_dir,
                       avg_loss / config.num_batches))


        reader.reset_batches()
        _classes = ['java.util', 'android.app', 'android.view', 'android.widget', 'java.io', 'javax.xml', 'java.net', \
        'android.graphics', 'android.content', 'android.webkit']
        _psi_encoders = []
        _labels = []
        _rev_dict = {v: k for k, v in config.decoder.vocab.items()}

        for b in range(config.num_batches):
            # setup the feed dict
            ev_data, n, e, y = reader.next_batch()
            feed = {model.targets: y}
            for j, ev in enumerate(config.evidence):
                feed[model.encoder.inputs[j].name] = ev_data[j]
            for j in range(config.decoder.max_ast_depth):
                feed[model.decoder.nodes[j].name] = n[j]
                feed[model.decoder.edges[j].name] = e[j]
                # Feeding value into reverse encoder
                feed[model.reverse_encoder.nodes[j].name] = n[j]
                feed[model.reverse_encoder.edges[j].name] = e[j]
            for arr in y:
                flag = 0
                for val in arr:
                    API_call = _rev_dict[val]
                    for _class in _classes:
                        if (API_call.find(_class)!= -1) == True:
                            flag = 1
                            _labels.append(_class)
                            break
                    if flag == 1:
                        break
                if flag == 0:
                    _labels.append('other_API')
            # run the optimizer
            _psi_encoder \
                = sess.run(model.psi_encoder, feed)

            _psi_encoders.append(_psi_encoder)

            _psi_encoders_agg = np.concatenate(_psi_encoders, axis = 0)


        embedding(_psi_encoders_agg,_labels)


def embedding(input_tensor, labels):
   metadata = os.path.join(LOG_DIR, 'metadata.tsv')

   images = tf.Variable( input_tensor , name='images')


   with open(metadata, 'w') as metadata_file:
       for row in range(input_tensor.shape[0]):
           c=labels[row]
           metadata_file.write('{}\n'.format(c))

   with tf.Session() as sess:
        saver = tf.train.Saver([images])

        sess.run(images.initializer)
        saver.save(sess, os.path.join(LOG_DIR, 'images.ckpt'))

        config = projector.ProjectorConfig()
        # One can add multiple embeddings.
        embedding = config.embeddings.add()
        embedding.tensor_name = images.name
        # Link this tensor to its metadata file (e.g. labels).
        embedding.metadata_path = metadata
        # Saves a config file that TensorBoard will read during startup.
        projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)



#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent(HELP))
    parser.add_argument('input_file', type=str, nargs=1,
                        help='input data file')
    parser.add_argument('--python_recursion_limit', type=int, default=10000,
                        help='set recursion limit for the Python interpreter')
    parser.add_argument('--save', type=str, default='save',
                        help='checkpoint model during training here')
    parser.add_argument('--config', type=str, default=None,
                        help='config file (see description above for help)')
    parser.add_argument('--continue_from', type=str, default=None,
                        help='ignore config options and continue training model checkpointed here')
    #clargs = parser.parse_args()
    clargs = parser.parse_args(['--config','config.json',
    # '..\..\..\..\..\..\data\DATA-training-top.json'])
    # '/home/rm38/Research/Bayou_Code_Search/bayou/data/DATA-training.json'])
        '/home/ubuntu/bayou/data/DATA-training.json'])
    sys.setrecursionlimit(clargs.python_recursion_limit)
    if clargs.config and clargs.continue_from:
        parser.error('Do not provide --config if you are continuing from checkpointed model')
    if not clargs.config and not clargs.continue_from:
        parser.error('Provide at least one option: --config or --continue_from')
    train(clargs)
