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

import argparse
import json
from collections import Counter
import ijson.backends.yajl2_cffi as ijson

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import re
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn import decomposition

from scripts.ast_extractor import get_ast_paths
from bayou.models.low_level_evidences.predict import BayesianPredictor
from bayou.models.low_level_evidences.utils import read_config
from bayou.experiments.predictMethods.SearchDB.utils import *

from scipy.cluster.vq import kmeans2


def load_desires():
    print("Loading API Dictionary")
    dict_api_calls = get_api_dict()
    print("Loading AST Dictionary")
    dict_ast = None #get_ast_dict()
    print("Loading Seq Dictionary")
    dict_seq = None #get_sequence_dict()
    return dict_api_calls, dict_ast, dict_seq


def main(clargs):
    sess = tf.InteractiveSession()
    predictor = BayesianPredictor(clargs.save, sess)
    print("model loaded")
    with open(os.path.join(clargs.save, 'config.json')) as f:
        config = read_config(json.load(f), chars_vocab=True)
    print("config read")

    dict_api_calls, dict_ast, dict_seq = load_desires()

    print('Reverse Encoder Plot')
    with open(clargs.input_file[0], 'rb') as f:
        call_k_means(f, 'b2', dict_api_calls)



def call_k_means(f, att, dict_api_calls, max_nums=1000):
    psis = []
    apis = []
    item_num = 0
    for program in ijson.items(f, 'programs.item'):
        if att not in program:
            return
        values = [float(val) for val in program[att]]
        psis.append(values)
        item_num += 1
        key = program['file'] + "/" + program['method']
        apis.append(dict_api_calls[key])
        if item_num > max_nums:
            break

    clusters, clustered_apis = k_means(psis, apis)

    sorted_ids = [i[0] for i in sorted(enumerate(clustered_apis), key=lambda x:get_intra_cluster_jaccards(x[1]), reverse=True)]

    clusters = [clusters[i] for i in sorted_ids]
    clustered_apis = [clustered_apis[i] for i in sorted_ids]


    # for j, clustered_apis_j in enumerate(clustered_apis):
    #     num_elems = print ('Number of elems in Cluster :: ' + str(j) + ' is :: ', len(clustered_apis_j))
    #     jac = get_intra_cluster_jaccards(clustered_apis_j)
    #     print ('Jaccard of Cluster :: ' + str(j) + ' is :: ', str(jac))

    num_clusters = len(clustered_apis)
    jac_matrix = np.zeros((num_clusters, num_clusters))
    for j, clustered_apis_j in enumerate(clustered_apis):
        print(j, end='.')
        for k, clustered_apis_k in enumerate(clustered_apis):
            jac = get_inter_cluster_jaccards(clustered_apis_j, clustered_apis_k)
            jac_matrix[j][k] = jac

    print('', end='[')
    for j in range(num_clusters):
        print('', end='[')
        for k in range(num_clusters):
            if k == num_clusters - 1:
                print("%.3f" % jac_matrix[j][k], end='],\n')
            else:
                print("%.3f" % jac_matrix[j][k], end=',')
    print(']')
    return


def k_means(psis, apis, num_centroids = 10):

    centroid, labels = kmeans2(np.array(psis), num_centroids)
    clusters = [[] for _ in range(num_centroids)]
    clustered_apis = [[] for _ in range(num_centroids)]
    for k, label in enumerate(labels):
        clusters[label].append(psis[k])
        clustered_apis[label].append(apis[k])
    return clusters, clustered_apis


def get_intra_cluster_jaccards(clustered_apis_k):

    dis_i = 0.
    for api_i in clustered_apis_k:
        for api_j in clustered_apis_k:
            dis_i += get_jaccard_distace_api(api_i, api_j)

    num_items = len(clustered_apis_k)
    return dis_i / (num_items * (num_items-1))


def get_inter_cluster_jaccards(clustered_apis_j, clustered_apis_k):

    dis_ = 0.
    for api_i in clustered_apis_j:
        for api_j in clustered_apis_k:
            dis_ += get_jaccard_distace_api(api_i, api_j)

    num_items_1 = len(clustered_apis_j)
    num_items_2 = len(clustered_apis_k)

    return dis_ / (num_items_1 * num_items_2 )

# def fitTSEandplot(psis, labels, name):
#     model = TSNE(n_components=2, init='random')
#     psis_2d = model.fit_transform(psis)
#
#     #pca = decomposition.PCA(n_components=2)
#     #pca.fit(psis)
#     #psis_2d = pca.transform(psis)
#
#     assert len(psis_2d) == len(labels)
#     scatter(clargs, zip(psis_2d, labels), name)
#
#
#
# def scatter(clargs, data, name):
#     dic = {}
#     for psi_2d, label in data:
#         if label == 'N/A':
#             continue
#         if label not in dic:
#             dic[label] = []
#         dic[label].append(psi_2d)
#
#     labels = list(dic.keys())
#     labels.sort(key=lambda l: len(dic[l]), reverse=True)
#     for label in labels[clargs.top:]:
#         del dic[label]
#
#
#     labels = dic.keys()
#     colors = cm.rainbow(np.linspace(0, 1, len(dic)))
#     plotpoints = []
#     for label, color in zip(labels, colors):
#         x = list(map(lambda s: s[0], dic[label]))
#         y = list(map(lambda s: s[1], dic[label]))
#         plotpoints.append(plt.scatter(x, y, color=color))
#
#     plt.legend(plotpoints, labels, scatterpoints=1, loc='lower left', ncol=3, fontsize=12)
#     plt.axhline(0, color='black')
#     plt.axvline(0, color='black')
#     plt.savefig(os.path.join(os.getcwd(), "plots/tSNE_" + name + ".jpeg"), bbox_inches='tight')
#     # plt.show()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_file', type=str, nargs=1,
                        help='input data file')
    parser.add_argument('--save', type=str, default='save',
                        help='directory to load model from')
    parser.add_argument('--top', type=int, default=10,
                        help='plot only the top-k labels')
    clargs = parser.parse_args()
    main(clargs)
