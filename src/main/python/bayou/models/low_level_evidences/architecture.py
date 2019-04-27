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

import tensorflow as tf
from itertools import chain
from bayou.models.low_level_evidences.seqEncoder import seqEncoder
from bayou.models.low_level_evidences.TreeEncoder import TreeEncoder
from bayou.models.low_level_evidences.TreeDecoder import TreeDecoder

class BayesianEncoder(object):
    def __init__(self, config, inputs, infer=False):

        # exists  = #ev * batch_size
        exists = [ev.exists(i, config, infer) for ev, i in zip(config.evidence, inputs)]
        zeros = tf.zeros([config.batch_size, config.latent_size], dtype=tf.float32)

        # Compute the denominator used for mean and covariance
        for ev in config.evidence:
            ev.init_sigma(config)

        d = [tf.where(exist, tf.tile([1. / tf.square(ev.sigma)], [config.batch_size]),
                      tf.zeros(config.batch_size)) for ev, exist in zip(config.evidence, exists)]
        d = 1. + tf.reduce_sum(tf.stack(d), axis=0)
        denom = tf.tile(tf.reshape(d, [-1, 1]), [1, config.latent_size])

        # Compute the mean of Psi
        with tf.variable_scope('mean'):
            # 1. compute encoding

            encodings = [ev.encode(i, config, infer) for ev, i in zip(config.evidence, inputs)]
            encodings = [encoding / tf.square(ev.sigma) for ev, encoding in
                         zip(config.evidence, encodings)]

            # 2. pick only encodings from valid inputs that exist, otherwise pick zero encoding
            self.encodings = [tf.where(exist, enc, zeros) for exist, enc in zip(exists, encodings)]

            # 3. tile the encodings according to each evidence type
            encodings = [[enc] * ev.tile for ev, enc in zip(config.evidence, self.encodings)]
            encodings = tf.stack(list(chain.from_iterable(encodings)))

            # 4. compute the mean of non-zero encodings
            self.psi_mean = tf.reduce_sum(encodings, axis=0) / denom

        # Compute the covariance of Psi
        with tf.variable_scope('covariance'):
            I = tf.ones([config.batch_size, config.latent_size], dtype=tf.float32)
            self.psi_covariance = I / denom


class BayesianDecoder(object):
    def __init__(self, config, initial_state, nodes, edges):

        emb = tf.get_variable('emb', [config.decoder.vocab_size, config.decoder.units])
        self.outputs = TreeDecoder( config, nodes, edges, initial_state, emb).all_outputs

        with tf.variable_scope("projections"):
            self.projection_w = tf.get_variable('projection_w', [config.decoder.units, config.decoder.vocab_size])
            self.projection_b = tf.get_variable('projection_b', [config.decoder.vocab_size])

        return




class SimpleDecoder(object):
    def __init__(self, config, emb, initial_state, nodes, ev_config):

        cell = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(ev_config.units)

        # projection matrices for output
        with tf.variable_scope("projection"):
            self.projection_w = tf.get_variable('projection_w', [ev_config.units,ev_config.vocab_size])
            self.projection_b= tf.get_variable('projection_b', [ev_config.vocab_size])


        def compute(i, cur_state, out):
            emb_inp = tf.nn.embedding_lookup(emb, nodes[i])
            output, cur_state = cell( emb_inp  , cur_state)
            return i+1, cur_state, out.write(i, output)

        time = tf.shape(nodes)[0]
        # setup embedding

        _, cur_state, out = tf.while_loop(
            lambda a, b, c: a < time,
            compute,
            (0, initial_state, tf.TensorArray(tf.float32, time)), parallel_iterations=1)

        self.outputs  =  tf.transpose(out.stack(), [1, 0, 2])

        return

class BayesianReverseEncoder(object):
    def __init__(self, config, nodes, edges, returnType, formalParam):

        embRT = tf.get_variable('emb_RT', [config.evidence[4].vocab_size, config.evidence[4].units])
        embFP = tf.get_variable('emb_FP', [config.evidence[5].vocab_size, config.evidence[5].units])

        with tf.variable_scope("Covariance"):
            with tf.variable_scope("APITree"):
                Tree_Cov = TreeEncoder(config, nodes, edges, config.latent_size).last_output

            with tf.variable_scope('ReturnType'):
                rt_Cov = seqEncoder( config.reverse_encoder.units, returnType, config.batch_size, config.latent_size, embRT).output

            with tf.variable_scope('FormalParam'):
                fp_Cov = seqEncoder(config.reverse_encoder.units, formalParam, config.batch_size, config.latent_size, embFP).output




        with tf.variable_scope("Mean"):
            with tf.variable_scope('APITree'):
                Tree_mean = TreeEncoder(config, nodes, edges,  config.latent_size).last_output

            with tf.variable_scope('ReturnType'):
                rt_mean = seqEncoder( config.reverse_encoder.units, returnType, config.batch_size, config.latent_size, embRT).output

            with tf.variable_scope('FormalParam'):
                fp_mean = seqEncoder(config.reverse_encoder.units, formalParam, config.batch_size, config.latent_size, embFP).output




            sigmas = [Tree_Cov , rt_Cov, fp_Cov]

            #dimension is  3*batch * 1
            finalSigma = tf.layers.dense(tf.reshape( tf.transpose(tf.stack(sigmas, axis=0), perm=[1,0,2]), [config.batch_size, -1]) , config.latent_size, activation=tf.nn.tanh)
            finalSigma = tf.layers.dense(finalSigma, config.latent_size, activation=tf.nn.tanh)

            finalSigma = tf.layers.dense(finalSigma, 1)

            d = tf.tile(tf.square(finalSigma),[1, config.latent_size])
            d = .00000001 + d
            self.psi_covariance = d

            encodings = [Tree_mean, rt_mean, fp_mean]
            finalMean = tf.layers.dense(tf.reshape( tf.transpose(tf.stack(encodings, axis=0), perm=[1,0,2]), [config.batch_size, -1]) , config.latent_size, activation=tf.nn.tanh)
            finalMean = tf.layers.dense(finalMean, config.latent_size, activation=tf.nn.tanh)
            finalMean = tf.layers.dense(finalMean, config.latent_size)
            # 4. compute the mean of non-zero encodings
            self.psi_mean = finalMean
