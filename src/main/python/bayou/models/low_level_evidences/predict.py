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
from bayou.models.low_level_evidences.utils import get_var_list, read_config

from bayou.models.low_level_evidences.architecture import BayesianEncoder


class BayesianPredictor(object):

    def __init__(self, save, sess):

        with open(os.path.join(save, 'config.json')) as f:
            config = read_config(json.load(f), chars_vocab=True)
        assert config.model == 'lle', 'Trying to load different model implementation: ' + config.model

        config.batch_size = 1
        self.config = config
        self.sess = sess
	
        self.inputs = [ev.placeholder(config) for ev in self.config.evidence]
        with tf.variable_scope("Encoder"):
            self.encoder = BayesianEncoder(config, self.inputs)
	

        self.EncA, self.EncB = self.calculate_ab(self.encoder.psi_mean , self.encoder.psi_covariance)

        # restore the saved model
        tf.global_variables_initializer().run()
        encoder_vars = get_var_list()['encoder_vars']
        saver = tf.train.Saver(encoder_vars)

        ckpt = tf.train.get_checkpoint_state(save)
        saver.restore(self.sess, ckpt.model_checkpoint_path)


    def get_a1b1(self, evidences):
        # setup initial states and feed
        # read and wrangle (with batch_size 1) the data
        inputs = [ev.wrangle([ev.read_data_point(evidences, infer=True)]) for ev in self.config.evidence]
        # setup initial states and feed
        feed = {}
        for j, ev in enumerate(self.config.evidence):
            feed[self.inputs[j].name] = inputs[j]
        [EncA, EncB] = self.sess.run( [ self.EncA, self.EncB ] , feed )
        return EncA, EncB

    def calculate_ab(self, mu, Sigma):
        a = -1 /(2*Sigma[:,0]) # slicing a so that a is now of shape (batch_size, 1)
        b = mu / Sigma
        return a, b
