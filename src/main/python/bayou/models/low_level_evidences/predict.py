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
from bayou.models.low_level_evidences.architecture import BayesianEncoder, BayesianReverseEncoder, BayesianDecoder, SimpleDecoder
from tensorflow.contrib import legacy_seq2seq as seq2seq
from bayou.models.low_level_evidences.node import Node, get_ast_from_json, CHILD_EDGE, SIBLING_EDGE, TooLongLoopingException, TooLongBranchingException

class BayesianPredictor(object):

    def __init__(self, save, sess):

        with open(os.path.join(save, 'config.json')) as f:
            config = read_config(json.load(f), chars_vocab=True)
        assert config.model == 'lle', 'Trying to load different model implementation: ' + config.model

        config.batch_size = 1
        self.config = config
        self.sess = sess

        infer = True

        self.inputs = [ev.placeholder(config) for ev in self.config.evidence]
        self.nodes = tf.placeholder(tf.int32, [config.batch_size, config.decoder.max_ast_depth])
        self.edges = tf.placeholder(tf.bool, [config.batch_size, config.decoder.max_ast_depth])
        self.targets = tf.placeholder(tf.int32, [config.batch_size, config.decoder.max_ast_depth])


        # targets  = tf.concat(  [self.nodes[:, 1:] , tf.zeros([config.batch_size , 1], dtype=tf.int32) ] ,axis=1 )  # shifted left by one

        ev_data = self.inputs[:-1]
        surr_input = self.inputs[-1][:-1]
        surr_input_fp = self.inputs[-1][-1]

        nodes = tf.transpose(self.nodes)
        edges = tf.transpose(self.edges)

        ###########################3
        with tf.variable_scope('Embedding'):
            emb = tf.get_variable('emb', [config.decoder.vocab_size, config.decoder.units])

        with tf.variable_scope("Encoder"):

            self.encoder = BayesianEncoder(config, ev_data, surr_input, surr_input_fp, infer)
            samples_1 = tf.random_normal([config.batch_size, config.latent_size], mean=0., stddev=1., dtype=tf.float32)

            self.psi_encoder = self.encoder.psi_mean + tf.sqrt(self.encoder.psi_covariance) * samples_1

        # setup the reverse encoder.
        with tf.variable_scope("Reverse_Encoder"):
            embAPI = tf.get_variable('embAPI', [config.reverse_encoder.vocab_size, config.reverse_encoder.units])
            embRT = tf.get_variable('embRT', [config.evidence[4].vocab_size, config.reverse_encoder.units])
            embFS = tf.get_variable('embFS', [config.evidence[5].vocab_size, config.reverse_encoder.units])
            self.reverse_encoder = BayesianReverseEncoder(config, embAPI, nodes, edges, ev_data[4], embRT, ev_data[5], embFS)
            samples_2 = tf.random_normal([config.batch_size, config.latent_size], mean=0., stddev=1., dtype=tf.float32)

            self.psi_reverse_encoder = self.reverse_encoder.psi_mean + tf.sqrt(self.reverse_encoder.psi_covariance) * samples_2

        # setup the decoder with psi as the initial state
        with tf.variable_scope("Decoder"):
            lift_w = tf.get_variable('lift_w', [config.latent_size, config.decoder.units])
            lift_b = tf.get_variable('lift_b', [config.decoder.units])
            initial_state = tf.nn.xw_plus_b(self.psi_encoder, lift_w, lift_b, name="Initial_State")
            self.decoder = BayesianDecoder(config, emb, initial_state, nodes, edges)

        with tf.variable_scope("RE_Decoder"):
            ## RE

            emb_RE = config.evidence[4].emb * 0.0 #tf.get_variable('emb_RE', [config.evidence[4].vocab_size, config.evidence[4].units])

            lift_w_RE = tf.get_variable('lift_w_RE', [config.latent_size, config.evidence[4].units])
            lift_b_RE = tf.get_variable('lift_b_RE', [config.evidence[4].units])

            initial_state_RE = tf.nn.xw_plus_b(self.psi_encoder, lift_w_RE, lift_b_RE, name="Initial_State_RE")

            input_RE = tf.transpose(tf.reverse_v2(tf.zeros_like(ev_data[4]), axis=[1]))
            output = SimpleDecoder(config, emb_RE, initial_state_RE, input_RE, config.evidence[4])

            projection_w_RE = tf.get_variable('projection_w_RE', [config.evidence[4].units, config.evidence[4].vocab_size])
            projection_b_RE = tf.get_variable('projection_b_RE', [config.evidence[4].vocab_size])
            logits_RE = tf.nn.xw_plus_b(output.outputs[-1] , projection_w_RE, projection_b_RE)

            labels_RE = tf.one_hot(tf.squeeze(ev_data[4]) , config.evidence[4].vocab_size , dtype=tf.int32)
            loss_RE = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_RE, logits=logits_RE)

            cond = tf.not_equal(tf.reduce_sum(self.encoder.psi_mean, axis=1), 0)
            # cond = tf.reshape( tf.tile(tf.expand_dims(cond, axis=1) , [1,config.evidence[5].max_depth]) , [-1] )
            self.loss_RE = tf.reduce_mean(tf.where(cond , loss_RE, tf.zeros(cond.shape)))

        with tf.variable_scope("FS_Decoder"):
            #FS
            emb_FS = config.evidence[5].emb #tf.get_variable('emb_FS', [config.evidence[5].vocab_size, config.evidence[5].units])
            lift_w_FS = tf.get_variable('lift_w_FS', [config.latent_size, config.evidence[5].units])
            lift_b_FS = tf.get_variable('lift_b_FS', [config.evidence[5].units])

            initial_state_FS = tf.nn.xw_plus_b(self.psi_encoder, lift_w_FS, lift_b_FS, name="Initial_State_FS")

            input_FS = tf.transpose(tf.reverse_v2(ev_data[5], axis=[1]))
            self.decoder_FS = SimpleDecoder(config, emb_FS, initial_state_FS, input_FS, config.evidence[5])

            output = tf.reshape(tf.concat(self.decoder_FS.outputs, 1), [-1, self.decoder_FS.cell1.output_size])
            logits_FS = tf.matmul(output, self.decoder_FS.projection_w_FS) + self.decoder_FS.projection_b_FS


            # logits_FS = output
            targets_FS = tf.reverse_v2(tf.concat( [ tf.zeros_like(ev_data[5][:,-1:]) , ev_data[5][:, :-1]], axis=1) , axis=[1])


            # self.gen_loss_FS = tf.contrib.seq2seq.sequence_loss(logits_FS, target_FS,
            #                                       tf.ones_like(target_FS, dtype=tf.float32))
            cond = tf.not_equal(tf.reduce_sum(self.encoder.psi_mean, axis=1), 0)
            cond = tf.reshape( tf.tile(tf.expand_dims(cond, axis=1) , [1,config.evidence[5].max_depth]) , [-1] )
            cond =tf.where(cond , tf.ones(cond.shape), tf.zeros(cond.shape))


            self.gen_loss_FS = seq2seq.sequence_loss([logits_FS], [tf.reshape(targets_FS, [-1])],
                                                  [cond])

        # get the decoder outputs
        with tf.name_scope("Loss"):
            output = tf.reshape(tf.concat(self.decoder.outputs, 1),
                                [-1, self.decoder.cell1.output_size])
            logits = tf.matmul(output, self.decoder.projection_w) + self.decoder.projection_b
            ln_probs = tf.nn.log_softmax(logits)


            # 1. generation loss: log P(Y | Z)
            cond = tf.not_equal(tf.reduce_sum(self.encoder.psi_mean, axis=1), 0)
            cond = tf.reshape( tf.tile(tf.expand_dims(cond, axis=1) , [1,config.decoder.max_ast_depth]) , [-1] )
            cond = tf.where(cond , tf.ones(cond.shape), tf.zeros(cond.shape))


            self.gen_loss = seq2seq.sequence_loss([logits], [tf.reshape(self.targets, [-1])], [cond])


            #KL_cond = tf.not_equal(tf.reduce_sum(self.encoder.psi_mean, axis=1) , 0)

            self.loss = self.gen_loss + 1/8 * self.loss_RE  + 8/8 * self.gen_loss_FS


        self.probY =  -1 * self.loss*8 + \
                             self.get_multinormal_lnprob(self.psi_encoder) - \
                             self.get_multinormal_lnprob(self.psi_encoder,self.encoder.psi_mean,self.encoder.psi_covariance)
        self.EncA, self.EncB = self.calculate_ab(self.encoder.psi_mean , self.encoder.psi_covariance)
        self.RevEncA, self.RevEncB = self.calculate_ab(self.reverse_encoder.psi_mean , self.reverse_encoder.psi_covariance)

        ## not required
        self.probYgivenX =  -1 * self.loss #\
                    #+ 1/256*(self.get_multinormal_lnprob(self.psi_encoder,self.encoder.psi_mean,self.encoder.psi_covariance) \
                    #- self.get_multinormal_lnprob(self.psi_encoder))
        ###############################



        # restore the saved model
        tf.global_variables_initializer().run()
        all_vars = tf.global_variables()
        saver = tf.train.Saver(all_vars)

        ckpt = tf.train.get_checkpoint_state(save)
        saver.restore(self.sess, ckpt.model_checkpoint_path)

        return




    def get_a1b1(self, evidences):

        inputs = self.wrange_inputs(evidences)


        feed = {}
        for j, _ in enumerate(self.config.evidence[:-1]):
            feed[self.inputs[j].name] = inputs[j]

        for j, _ in enumerate(self.config.evidence[-1].internal_evidences[:-1]):
            feed[self.inputs[-1][j].name] = inputs[-1][j]

        for j in range(2): #len(self.config.evidence[-1].internal_evidences[-1])):
            feed[self.inputs[-1][-1][j].name] = inputs[-1][-1][j]


        [EncA, EncB] = self.sess.run( [ self.EncA, self.EncB ] , feed )
        return EncA, EncB


    def wrange_inputs(self, program):

        rdp = [ev.read_data_point(program, infer=True) for ev in self.config.evidence]

        config = self.config
        raw_evidences = [rdp for j in range(self.config.batch_size)]
        raw_evidences = [[raw_evidence[i] for raw_evidence in raw_evidences] for i, ev in enumerate(config.evidence)]
        raw_evidences[-1] = [[raw_evidence[j] for raw_evidence in raw_evidences[-1]] for j in range(len(config.surrounding_evidence))] # for
        raw_evidences[-1][-1] = [[raw_evidence[j] for raw_evidence in raw_evidences[-1][-1]] for j in range(2)] # is
        rdp = raw_evidences

        # inputs = [ev.wrangle([ev_rdp for i in range(self.config.batch_size)]) for ev, ev_rdp in zip(self.config.evidence, rdp)]
        inputs = [ev.wrangle(data) for ev, data in zip(config.evidence, rdp)]

        return inputs


    def wrangle_data(self, program):

        inputs = self.wrange_inputs(program)

        nodes = np.zeros((self.config.batch_size, self.config.decoder.max_ast_depth), dtype=np.int32)
        edges = np.zeros((self.config.batch_size, self.config.decoder.max_ast_depth), dtype=np.bool)
        targets = np.zeros((self.config.batch_size, self.config.decoder.max_ast_depth), dtype=np.int32)

        ast_node_graph = get_ast_from_json(program['ast']['_nodes'])
        path = ast_node_graph.depth_first_search()

        # parsed_data_array = []
        for i, (curr_node_val, parent_node_id, edge_type) in enumerate(path):
            try:
                curr_node_id = self.config.decoder.vocab[curr_node_val]
                parent_call = path[parent_node_id][0]
                parent_call_id = self.config.decoder.vocab[parent_call]

                if i > 0: # and not (curr_node_id is None or parent_call_id is None): # I = 0 denotes DSubtree ----sibling---> DSubTree
                    nodes[0,i-1] = parent_call_id
                    edges[0,i-1] = edge_type
                    targets[0,i-1] = curr_node_id
            except:
                pass

        return nodes, edges, targets, inputs

    def get_a1b1a2b2(self, program):

        nodes, edges, targets, inputs = self.wrangle_data(program)
        ignored = True if np.sum(nodes)==0 else False

        feed = {}
        for j, _ in enumerate(self.config.evidence[:-1]):
            feed[self.inputs[j].name] = inputs[j]

        for j, _ in enumerate(self.config.evidence[-1].internal_evidences[:-1]):
            feed[self.inputs[-1][j].name] = inputs[-1][j]

        for j in range(2): #len(self.config.evidence[-1].internal_evidences[-1])):
            feed[self.inputs[-1][-1][j].name] = inputs[-1][-1][j]

        feed[self.nodes.name] = nodes
        feed[self.edges.name] = edges
        feed[self.targets.name] = targets

        [EncA, EncB, RevEncA, RevEncB, probY] = self.sess.run( [  self.EncA, self.EncB , self.RevEncA, self.RevEncB , self.probY ] , feed )
        return EncA, EncB, RevEncA, RevEncB, probY, ignored



    def get_probYgivenX(self, nodes, edges, targets, inputs):

        feed = {}
        for j, _ in enumerate(self.config.evidence[:-1]):
            feed[self.inputs[j].name] = inputs[j]

        for j, _ in enumerate(self.config.evidence[-1].internal_evidences[:-1]):
            feed[self.inputs[-1][j].name] = inputs[-1][j]

        for j in range(2): #len(self.config.evidence[-1].internal_evidences[-1])):
            feed[self.inputs[-1][-1][j].name] = inputs[-1][-1][j]

        feed[self.nodes.name] = nodes
        feed[self.edges.name] = edges
        feed[self.targets.name] = targets

        [probYgivenX] = self.sess.run( [ self.probYgivenX ], feed)
        return  probYgivenX



    def get_ev_sigma(self, evidences):
        # setup initial states and feed
        # read and wrangle (with batch_size 1) the data
        inputs = [ev.wrangle([ev.read_data_point(evidences, infer=True)]) for ev in self.config.evidence]
        # setup initial states and feed
        feed = {}
        for j, ev in enumerate(self.config.evidence):
            feed[self.inputs[j].name] = inputs[j]
        allEvSigmas = self.sess.run( [ ev.sigma for ev in self.config.evidence ] , feed )
        return allEvSigmas

    def calculate_ab(self, mu, Sigma):
        a = -1 /(2*Sigma[:,0]) # slicing a so that a is now of shape (batch_size, 1)
        b = mu / Sigma
        return a, b

    def get_multinormal_lnprob(self, x, mu=None , Sigma=None ):
        if mu is None:
            mu = tf.zeros(x.shape)
        if Sigma is None:
            Sigma = tf.ones(x.shape)

        # mu is a vector of size [batch_size, latent_size]
        #sigma is another vector of size [batch_size, latent size] denoting a diagonl matrix
        ln_nume =  -0.5 * tf.reduce_sum( tf.square(x-mu) / Sigma, axis=1 )
        ln_deno = self.config.latent_size / 2 * tf.log(2 * np.pi ) + 0.5 * tf.reduce_sum(tf.log(Sigma), axis=1)
        val = ln_nume - ln_deno

        return val
