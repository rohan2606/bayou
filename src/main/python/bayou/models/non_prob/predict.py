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
from bayou.models.non_prob.architecture import BayesianEncoder, BayesianReverseEncoder

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


        with tf.variable_scope("Encoder"):
            self.encoder = BayesianEncoder(config, ev_data, surr_input, surr_input_fp, infer)
            self.psi_encoder = self.encoder.psi_mean

        # setup the reverse encoder.
        with tf.variable_scope("Reverse_Encoder", reuse=tf.AUTO_REUSE):
            embAPI = tf.get_variable('embAPI', [config.reverse_encoder.vocab_size, config.reverse_encoder.units])
            embRT = tf.get_variable('embRT', [config.evidence[4].vocab_size, config.reverse_encoder.units])
            embFS = tf.get_variable('embFS', [config.evidence[5].vocab_size, config.reverse_encoder.units])
            self.reverse_encoder = BayesianReverseEncoder(config, embAPI, nodes, edges,  ev_data[4], embRT, ev_data[5], embFS)
            self.psi_reverse_encoder = self.reverse_encoder.psi_mean


            # 1. generation loss: log P(Y | Z)

            self.positive_distance = self.cosine_similarity(self.psi_encoder, self.psi_reverse_encoder) # - self.cosine_similarity(self.psi_encoder, self.psi_reverse_encoder_negative)


        ###############################



        # restore the saved model
        tf.global_variables_initializer().run()
        all_vars = tf.global_variables()
        saver = tf.train.Saver(all_vars)

        ckpt = tf.train.get_checkpoint_state(save)
        saver.restore(self.sess, ckpt.model_checkpoint_path)

        return






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

    def get_psi_encoder(self, program):

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

        [psi_enc] = self.sess.run( [ self.psi_encoder  ] , feed )
        return psi_enc



    def get_psi_rev_enc(self, program):

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

        [psi_rev_enc] = self.sess.run( [ self.psi_reverse_encoder  ] , feed )
        return psi_rev_enc


    def cosine_similarity(self, a, b):
       norm_a = tf.nn.l2_normalize(a,0)
       norm_b = tf.nn.l2_normalize(b,0)
       return 1 - tf.reduce_sum(tf.multiply(norm_a, norm_b))
