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
from bayou.models.low_level_evidences.utils import lse


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

    def get_lnProb_Y_i(self, evidences, nodes, edges, targets, num_psi_samples=10):
        """

        :param evidences: the input evidences
        :param num_psi_samples: number of samples of the intent, averaged before AST construction
        :return: probabilities
        """
        probs = []
        for i in range(num_psi_samples):
            psi, psi_mean, psi_Sigma = self.psi_from_evidence(evidences)
            # the prob that we get here is the P(Y|Z) where Z~P(Z|X). It still needs to multiplied by P(Z)/P(Z|X) to get the correct value
            prob = self.model.infer_lnprobY_given_psi(self.sess, psi, nodes, edges, targets)
            #prob is now scalar but in ideal case it will be [batch_size]
            prob +=  self.get_psi_lnprob(psi) - self.get_psi_lnprob(psi, psi_mean, psi_Sigma)
            #also get_psi_lnprob is of size [batch_size] so they should add up.
            probs.append(prob)
        
        # probs should be concatenated when batch is used
        # probs = np.transpose(np.array(probs))
        
        #This is wrong, need to do LSE trick
        # in batch case probs is [batch_size , num_psi_samples]
        avg_prob = lse(probs)
        return avg_prob

    def get_psi_lnprob(self, x, mu=None , sigma=None ):
        
        if mu is None:
            mu = np.zeros(x.shape)
        if sigma is None:
            sigma = np.ones(x.shape)
        
        # mu is a vector of size [batch_size, latent_size]
        #sigma is another vector of size [batch_size, latent size] denoting a diagonl matrix
        ln_nume =  -0.5 * np.sum( np.square(x-mu) /sigma, axis=1 )  
        ln_deno = len(x)/2 * np.log(2 * np.pi ) + 0.5 * np.sum(np.log(sigma), axis=1)
        val = ln_nume - ln_deno
        
        return val[0] # take the first batch

    def psi_from_evidence(self, js_evidences):
        """
        Gets a latent intent from the model, given some evidences

        :param js_evidences: the evidences
        :return: the latent intent
        """
        psi, psi_mean, psi_Sigma = self.model.infer_psi_encoder(self.sess, js_evidences)
        return psi, psi_mean, psi_Sigma

    def psi_from_output(self, nodes, edges, js_evidences):
        """
        """
        psi_re, psi_re_mu, psi_re_sigma = self.model.infer_psi_reverse_encoder(self.sess, nodes, edges, js_evidences)
        return psi_re, psi_re_mu, psi_re_sigma


    def get_encoder_ab(self, evidences):
        psi_e, psi_e_mu, psi_e_Sigma = self.psi_from_evidence(evidences)
        a1, b1 = self.calculate_ab(psi_e_mu, psi_e_Sigma)
        return a1, b1

    def get_rev_encoder_ab(self, nodes, edges, js_evidences):
        psi_re, psi_re_mu, psi_re_sigma = self.psi_from_output(nodes, edges, js_evidences)
        a2, b2= self.calculate_ab(psi_re_mu, psi_re_sigma)
        return a2, b2

    def calculate_ab(self, mu, Sigma):
        a = -1 /(2*Sigma)
        b = mu / Sigma
        return a, b
    
    def get_c_minus_cstar(self, a1,b1, a2, b2):
        """
        """
        
        t1 = np.sum(np.square(b1)/(4*a1)) + np.sum(np.square(b2)/(4*a2))
        t2 = -0.5 * np.sum(np.log(-1/(2*a1))) -0.5 * np.sum(np.log(-1/(2*a2)))
        t3 = - (3/2) * len(a1)* np.log(2*np.pi)
        
        c = t1+t2+t3
        
        b_star = b1 + b2
        a_star = a1 + a2 + 0.5
        
        t1_star = np.sum(np.square(b_star)/(4*a_star))
        t2_star = -0.5 * np.sum(np.log(-1/(2*a_star)))
        t3_star = - (1/2) * len(a1)* np.log(2*np.pi)
        
        c_star = t1_star + t2_star + t3_star
        
        
        prob = (c - c_star)
        return prob
