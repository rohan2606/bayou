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
from bayou.models.low_level_evidences.gru_tree import TreeEncoder
from bayou.models.low_level_evidences.seqEncoder import seqEncoder

class BayesianEncoder(object):
    def __init__(self, config, inputs, surr_input, surr_input_fp, infer=False):

        # exists  = #ev * batch_size
        exists = [ev.exists(i, config, infer) for ev, i in zip(config.evidence[:-1], inputs)]

        surr_input = list(surr_input)
        surr_input.append(surr_input_fp)
        surr_input_new = tuple(surr_input)
         
        for ev in config.evidence:
           ev.init_sigma(config)

        exists.append(config.evidence[-1].exists(surr_input_new, config, infer))
        zeros = tf.zeros([config.batch_size, config.latent_size], dtype=tf.float32)


        # Compute the mean of Psi
        with tf.variable_scope('mean'):
            # 1. compute encoding

            encodings = [ev.encode(i, config, infer) for ev, i in zip(config.evidence[:-1], inputs)]
            encodings.append(config.evidence[-1].encode(surr_input_new, config, infer))


            # 2. pick only encodings from valid inputs that exist, otherwise pick zero encoding
            encodings = [tf.where(exist, enc, zeros) for exist, enc in zip(exists, encodings)]

            # # 3. tile the encodings according to each evidence type

            # 4. compute the mean of non-zero encodings
            #self.psi_mean = tf.reduce_sum(encodings, axis=0)
            self.psi_mean = tf.layers.dense(tf.concat(encodings, axis=1),config.latent_size,activation=tf.nn.tanh)
 



class BayesianReverseEncoder(object):
    def __init__(self, config, emb, nodes, edges, returnType, embRE, formalParam, embFP):

        nodes = [ nodes[i] for i in range(config.reverse_encoder.max_ast_depth)]
        edges = [ edges[i] for i in range(config.reverse_encoder.max_ast_depth)]


        with tf.variable_scope('APITree'):
            API_Mean_Tree = TreeEncoder(emb, config.batch_size, nodes, edges, config.reverse_encoder.num_layers, \
                                config.reverse_encoder.units, config.reverse_encoder.max_ast_depth, config.latent_size)
            Tree_mean = API_Mean_Tree.last_output

        with tf.variable_scope('ReturnType'):
            Ret_Seq = seqEncoder(config.reverse_encoder.num_layers, config.reverse_encoder.units, returnType, config.batch_size, embRE, config.latent_size)

            w = tf.get_variable('w', [config.reverse_encoder.units, config.latent_size])
            b = tf.get_variable('b', [config.latent_size])

            rt_mean = tf.nn.xw_plus_b(Ret_Seq.output ,w, b)


        with tf.variable_scope('FormalParam'):
            fp_Seq = seqEncoder(config.reverse_encoder.num_layers, config.reverse_encoder.units, formalParam, config.batch_size, embFP, config.latent_size)

            w = tf.get_variable('w', [config.reverse_encoder.units, config.latent_size])
            b = tf.get_variable('b', [config.latent_size])

            fp_mean = tf.nn.xw_plus_b(fp_Seq.output,w, b)


        encodings = [Tree_mean, rt_mean, fp_mean]
        finalMean = tf.layers.dense(tf.reshape( tf.transpose(tf.stack(encodings, axis=0), perm=[1,0,2]), [config.batch_size, -1]) , config.latent_size, activation=tf.nn.tanh)
        finalMean = tf.layers.dense(finalMean, config.latent_size, activation=tf.nn.tanh)
        finalMean = tf.layers.dense(finalMean, config.latent_size, activation=tf.nn.tanh)
        # 4. compute the mean of non-zero encodings
        self.psi_mean = finalMean
