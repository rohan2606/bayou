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

    def get_lnProbY(self, evidences, nodes, edges, targets, num_psi_samples=1):
        """
        :param evidences: the input evidences
        :param num_psi_samples: number of samples of the intent, averaged before AST construction
        :return: probabilities
        """
        probs = []
        for i in range(num_psi_samples):
            psi, psi_mean, psi_Sigma = self.model.infer_psi_encoder(self.sess, evidences)
            # the prob that we get here is the P(Y|Z) where Z~P(Z|X). It still needs to multiplied by P(Z)/P(Z|X) to get the correct value
            prob = self.model.infer_lnprobY_given_psi(self.sess, psi, nodes, edges, targets)
            prob =  prob + self.get_multinormal_prob(psi) - self.get_multinormal_prob(psi, psi_mean, psi_Sigma)
            #also get_multinormal_prob is of size [batch_size] so they should add up.
            probs.append(prob)

        probs = np.transpose(np.stack(probs, axis=0))
        # in batch case probs is [batch_size , num_psi_samples]
        avg_prob = get_sum_in_log(probs) - np.log(num_psi_samples)
        return avg_prob # np array of size batch_size

    def get_multinormal_prob(self, x, mu=None , Sigma=None ):

        if mu is None:
            mu = np.zeros(x.shape)
        if Sigma is None:
            Sigma = np.ones(x.shape)

        # mu is a vector of size [batch_size, latent_size]
        #sigma is another vector of size [batch_size, latent size] denoting a diagonl matrix
        ln_nume =  -0.5 * np.sum( np.square(x-mu) / Sigma, axis=1 )
        ln_deno = x.shape[1]/2 * np.log(2 * np.pi ) + 0.5 * np.sum(np.log(Sigma), axis=1)
        val = ln_nume - ln_deno

        return val


    def get_encoder_ab(self, evidences):
        psi_e, psi_e_mu, psi_e_Sigma = self.model.infer_psi_encoder(self.sess, evidences)
        a1, b1 = self.calculate_ab(psi_e_mu, psi_e_Sigma)
        return a1, b1

    def get_rev_encoder_ab(self, nodes, edges, evidences):
        psi_re, psi_re_mu, psi_re_Sigma = self.model.infer_psi_reverse_encoder(self.sess, nodes, edges, evidences)
        a2, b2= self.calculate_ab(psi_re_mu, psi_re_Sigma)
        return a2, b2

    def calculate_ab(self, mu, Sigma):
        a = -1 /(2*Sigma)
        b = mu / Sigma
        return a, b

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
