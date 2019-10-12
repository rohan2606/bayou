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

from bayou.models.non_prob.model import Model
from bayou.models.low_level_evidences.utils import get_var_list


class TooLongPathError(Exception):
    pass


class IncompletePathError(Exception):
    pass


class InvalidSketchError(Exception):
    pass


class BayesianPredictor(object):

    def __init__(self, save, sess, config, iterator):
        self.sess = sess
        self.model = Model(config, iterator, infer=True)
        self.config = config
        #
        # restore the saved model
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(save)
        saver.restore(self.sess, ckpt.model_checkpoint_path)


    def get_latent_vector(self):
        # setup initial states and feed
        [prog_psi] = self.sess.run([self.model.psi_reverse_encoder])

        return prog_psi
