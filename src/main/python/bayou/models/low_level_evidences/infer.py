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

    def get_c_minus_cstar(self, a1,b1, a2, b2, prob_Y):
        a_star = a1 + a2 + 0.5
        b_star = b1 + b2

        ab1 = self.get_contribs(a1, b1)
        ab2 = self.get_contribs(a2, b2)
        ab_star = self.get_contribs(a_star, b_star)
        cons = 0.5* self.model.config.latent_size * np.log( 2*np.pi )

        prob = ab1 + ab2 - ab_star - cons
        prob += np.array(prob_Y)
        return prob

    def get_contribs(self, a , b):
        assert(a.shape == b.shape)
        assert(len(list(a.shape)) <= 2)
        if (len(list(a.shape)) == 2):
            temp = np.sum(np.square(b)/(4*a), axis=1) + 0.5 * np.sum(np.log(-a/np.pi), axis=1)
        else:
            temp = np.sum(np.square(b)/(4*a), axis=0) + 0.5 * np.sum(np.log(-a/np.pi), axis=0)
        return temp
