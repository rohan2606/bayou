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
import json

from bayou.models.low_level_evidences.model import Model
from bayou.models.low_level_evidences.utils import CHILD_EDGE, SIBLING_EDGE
from bayou.models.low_level_evidences.utils import read_config

MAX_GEN_UNTIL_STOP = 20
MAX_AST_DEPTH = 5


class TooLongPathError(Exception):
    pass


class IncompletePathError(Exception):
    pass


class InvalidSketchError(Exception):
    pass


class BayesianPredictor(object):

    def __init__(self, save, sess, config):
        self.sess = sess


        self.model = Model(config, True)

        # load the callmap
        with open(os.path.join(save, 'callmap.pkl'), 'rb') as f:
            self.callmap = pickle.load(f)

        # restore the saved model
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(save)
        saver.restore(self.sess, ckpt.model_checkpoint_path)

    def get_Prob_Y_i(self, evidences, nodes, edges, targets, num_psi_samples=100):
        """

        :param evidences: the input evidences
        :param num_psi_samples: number of samples of the intent, averaged before AST construction
        :return: probabilities
        """
        probs = []
        for i in range(num_psi_samples):
            psi, psi_mean, psi_sigma = self.psi_from_evidence(evidences)
            # the prob that we get here is the P(Y|Z) where Z~P(Z|X). It still needs to multiplied by P(Z)/P(Z|X) to get the correct value
            prob = self.model.infer_probY_given_psi(self.sess, psi, nodes, edges, targets)
            prob = prob * self.get_psi_prob(psi) / self.get_psi_prob(psi, psi_mean, psi_sigma)
            probs.append(prob)

        avg_prob = np.mean(probs, axis=0)
        return avg_prob

    def get_psi_prob(self, x, mu=0, sigma=1):
        val = np.exp( -1*np.square(x-mu)/2*np.square(sigma) )/np.sqrt(2*np.pi*np.square(sigma))
        return val

    def psi_from_evidence(self, js_evidences):
        """
        Gets a latent intent from the model, given some evidences

        :param js_evidences: the evidences
        :return: the latent intent
        """
        psi, psi_mean, psi_sigma = self.model.infer_psi_encoder(self.sess, js_evidences)
        return psi, psi_mean, psi_sigma

    def psi_from_output(self, nodes, edges):
        """
        """
        psi_re, psi_re_mu, psi_re_sigma = self.model.infer_psi_reverse_encoder(self.sess, nodes, edges)
        return psi_re, psi_re_mu, psi_re_sigma


    def get_encoder_abc(self, evidences):
        psi_e, psi_e_mu, psi_e_sigma = self.psi_from_evidence(evidences)
        a1, b1, c1 = self.calculate_abc(psi_e_mu, psi_e_sigma)
        return [a1, b1, c1]

    def get_rev_encoder_abc(self, nodes, edges):
        psi_re, psi_re_mu, psi_re_sigma = self.psi_from_output(nodes, edges)
        a2, b2, c2 = self.calculate_abc(psi_re_mu, psi_re_sigma)
        return [a2,b2,c2]

    def get_PY_given_Xi(self, abc1, abc2):
        """
        """
        a1,b1,c1 = abc1
        a2,b2,c2 = abc2
        t1 = np.square(b1)/(4*a1) + np.square(b2)/(4*a2) - np.square(b1+b2)/(4*(a1+a2+0.5)) \
                    + 0.5*np.log(2*a1*a2/np.pi) - 0.5*np.log(-1*(a1+a2+0.5)/np.pi)
        prob = np.exp(t1)
        return prob

    def calculate_abc(self, mu, sigma):
        a = -1 /(2*np.square(sigma))
        b = mu / np.square(sigma)
        c = -1 * np.square(mu)/(2*np.square(sigma))
        return a, b, c
