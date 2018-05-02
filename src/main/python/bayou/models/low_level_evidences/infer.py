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
import tensorflow as tf
import numpy as np

import os
import pickle

from bayou.models.low_level_evidences.model import Model
from bayou.models.low_level_evidences.utils import get_sum_in_log, get_var_list


class TooLongPathError(Exception):
    pass


class IncompletePathError(Exception):
    pass


class InvalidSketchError(Exception):
    pass


class BayesianPredictor(object):

    def __init__(self, save, sess, config, bayou_mode=False):
        self.sess = sess


        self.model = Model(config, infer=True, bayou_mode=bayou_mode)

        # load the callmap
        with open(os.path.join(save, 'callmap.pkl'), 'rb') as f:
            self.callmap = pickle.load(f)

        # restore the saved model
        tf.global_variables_initializer().run()
        if bayou_mode == True:
            bayou_vars = get_var_list()['bayou_vars']
            saver = tf.train.Saver(bayou_vars)
        else:
            saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(save)
        saver.restore(self.sess, ckpt.model_checkpoint_path)


    def get_all_params_inago(self, evidences, nodes, edges, targets):
        # setup initial states and feed
        inputs = evidences
        feed = {self.model.targets: targets}
        for j,ev in enumerate(self.model.config.evidence):
            feed[self.model.encoder.inputs[j].name] = inputs[j]
        for j in range(self.model.config.decoder.max_ast_depth):
            feed[self.model.decoder.nodes[j].name] = nodes[j]
            feed[self.model.decoder.edges[j].name] = edges[j]
        for j in range(self.model.config.reverse_encoder.max_ast_depth):
            feed[self.model.reverse_encoder.nodes[j].name] = nodes[self.model.config.reverse_encoder.max_ast_depth - 1 - j]
            feed[self.model.reverse_encoder.edges[j].name] = edges[self.model.config.reverse_encoder.max_ast_depth - 1 - j]


        [probY, EncA, EncB, RevEncA, RevEncB] = self.sess.run([self.model.probY, self.model.EncA, self.model.EncB,\
                                                        self.model.RevEncA, self.model.RevEncB], feed)

        return probY, EncA, EncB, RevEncA, RevEncB

    def similarity(self, _a1, _b1, _a2, _b2, prob_Y):
            # a1 = tf.placeholder(tf.float32,[self.model.config.latent_size])
            a1_in = tf.placeholder(tf.float32,[])
            a1 = tf.tile(tf.reshape(a1_in,[1]),[self.model.config.latent_size])

            b1_in = tf.placeholder(tf.float32,[self.model.config.latent_size])
            b1 = b1_in
            # a2 = tf.placeholder(tf.float32,[self.model.config.batch_size,self.model.config.latent_size])
            a2_in = tf.placeholder(tf.float32,[self.model.config.batch_size])
            a2 = tf.tile(tf.expand_dims(a2_in,axis=1),[1,self.model.config.latent_size])

            b2_in = tf.placeholder(tf.float32,[self.model.config.batch_size,self.model.config.latent_size])
            b2 = b2_in
            t1 = tf.reduce_sum(tf.square(b1)/(4*a1), axis=0) + 0.5 * tf.reduce_sum(tf.log(-a1/np.pi), axis=0)
            t2 = tf.reduce_sum(tf.square(b2)/(4*a2), axis=1) + 0.5 * tf.reduce_sum(tf.log(-a2/np.pi), axis=1)
            t3 = 0.5 * self.model.config.latent_size * tf.log(2*np.pi)
            c = t1 + t2 - t3

            b_star = b1 + b2
            a_star = a1 + a2 + 0.5
            c_star = tf.reduce_sum(tf.square(b_star)/(4*a_star), axis=1) + 0.5 * tf.reduce_sum(tf.log(-a_star/np.pi), axis=1)
            prob = (c - c_star)

            _prob = self.sess.run(prob, feed_dict={a1_in:_a1, b1_in:_b1, a2_in:_a2, b2_in:_b2})

            _prob += np.array(prob_Y)
            return _prob
