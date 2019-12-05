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


from scipy.cluster.vq import kmeans2
from bayou.experiments.predictMethods.SearchDB.utils import get_api_dict,get_ast_dict,get_sequence_dict

def load_desires():
    print("Loading API Dictionary")
    dict_api_calls = get_api_dict()
    #print("Loading AST Dictionary")
    dict_ast = None #get_ast_dict()
    #print("Loading Seq Dictionary")
    dict_seq = None# get_sequence_dict()
    return dict_api_calls, dict_ast, dict_seq




def plotter(matrix, vector, name='temp'):
   
    num_centroids = len(vector) 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix, interpolation='nearest' , cmap=matplotlib.cm.get_cmap('Blues'))
    cax.set_clim(0.0, 0.5)
    cbar = fig.colorbar(cax)
    cbar.set_label('Average Jaccard Similarity', size=16)
    cbar.ax.tick_params(labelsize=16)

    xticks = list(range(num_centroids))
    yticks = list(range(num_centroids))

    ax.set_xticks(xticks)
    ax.set_xticklabels( [str(val+1) if (val+1)==1 or (val+1)%5==0 else '' for val in xticks] , fontsize=16)
    ax.set_yticks(yticks)
    ax.set_yticklabels( [str(val+1) if (val+1)==1 or (val+1)%5==0 else '' for val in yticks] , fontsize=16)
    ax.xaxis.set_ticks_position('bottom')


    plt.xlabel('Cluster Number',fontsize=16)
    plt.ylabel('Cluster Number',fontsize=16)
    plt.savefig(name + '.png')
    with open(name + '.json','w') as f:
         json.dump({'jaccard_intra_cluster':vector},f,indent=4)
    return
   


def main(clargs):
    
    if clargs.index == 'b2':
        from bayou.models.low_level_evidences.predict import BayesianPredictor
        from bayou.models.low_level_evidences.utils import read_config
    else:
        from bayou.models.non_prob.predict import BayesianPredictor
        from bayou.models.non_prob.utils import read_config
    
    num_centroids = 20

    sess = tf.InteractiveSession()
    predictor = BayesianPredictor(clargs.save, sess)
    print("model loaded")
    with open(os.path.join(clargs.save, 'config.json')) as f:
        config = read_config(json.load(f), chars_vocab=True)
    print("config read")
    dict_api_calls, dict_ast, dict_seq = load_desires()

    print('API Call Jaccard Calculations')
    with open(clargs.input_file[0], 'rb') as f:
        jac_api_matrix, jac_api_vector = call_k_means(f, clargs.index, dict_api_calls, num_centroids=num_centroids)

    plotter(jac_api_matrix, jac_api_vector, name='api_jaccard')

    #print('Seq Calls Jaccard Calculations')
    #with open(clargs.input_file[0], 'rb') as f:
    #    jac_seq_matrix, jac_seq_vector = call_k_means(f, clargs.index, dict_seq, num_centroids=num_centroids)

    #plotter(jac_seq_matrix, jac_seq_vector, name='seq_jaccard')

    #print('AST Jaccard Calculations')
    #with open(clargs.input_file[0], 'rb') as f:
    #    jac_ast_matrix = call_k_means(f, clargs.index, dict_ast)

    return


def call_k_means(f, att, dict_api_calls, max_nums=100000, num_centroids=10):
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

    clusters, clustered_apis = k_means(psis, apis, num_centroids=num_centroids)

    cluster_jaccards = [get_intra_cluster_jaccards(x) for x in clustered_apis]

    #print(sorted(cluster_jaccards, reverse=True))
    intra_cluster_jaccards = sorted(cluster_jaccards, reverse=True)
    sorted_ids = [i[0] for i in sorted(enumerate(cluster_jaccards), key=lambda x:x[1], reverse=True)]

    clusters = [clusters[i] for i in sorted_ids]
    clustered_apis = [clustered_apis[i] for i in sorted_ids]


    # for j, clustered_apis_j in enumerate(clustered_apis):
    #     num_elems = print ('Number of elems in Cluster :: ' + str(j) + ' is :: ', len(clustered_apis_j))
    #     jac = get_intra_cluster_jaccards(clustered_apis_j)
    #     print ('Jaccard of Cluster :: ' + str(j) + ' is :: ', str(jac))

    num_clusters = len(clustered_apis)
    jac_matrix = np.zeros((num_clusters, num_clusters))
    for j, clustered_apis_j in enumerate(clustered_apis):
        for k, clustered_apis_k in enumerate(clustered_apis):
            if k > j:
                jac = get_inter_cluster_jaccards(clustered_apis_j, clustered_apis_k)
                jac_matrix[j][k] = jac
            elif k == j:
                jac = get_intra_cluster_jaccards(clustered_apis_j)
                jac_matrix[j][k] = jac
            else:
                jac_matrix[j][k] = jac_matrix[k][j]

#    print('', end='[')
#    for j in range(num_clusters):
#        print('', end='[')
#        for k in range(num_clusters):
#            if k == num_clusters - 1 and j == num_clusters - 1:
#                print("%.3f" % jac_matrix[j][k], end=']\n')
#            elif k == num_clusters - 1:
#                print("%.3f" % jac_matrix[j][k], end='],\n')
#            else:
#                print("%.3f" % jac_matrix[j][k], end=',')
#    print(']')

    return jac_matrix,  intra_cluster_jaccards 


def get_jaccard_distace(a,b):
    if type(a) == str and type(eval(a)) == list :
        setA = set([tuple(item['calls']) for item in eval(a)])
        setB = set([tuple(item['calls']) for item in eval(b)])
    elif type(a) == str and type(eval(a)) == dict :
        dictA = eval(a.replace("u'", "'"))
        dictB = eval(b.replace("u'", "'"))
        return 1. if dictA == dictB else 0.
    elif type(a) == list:
        setA = set(a)
        setB = set(b)
    else:
        print(type(a))

    if (len(setA) == 0) and (len(setB) == 0):
        return 1

    distance = len(setA & setB) / len(setA | setB)
    return distance


def k_means(psis, apis, num_centroids = 20, max_cap=200):

    centroid, labels = kmeans2(np.array(psis), num_centroids)
    clusters = [[] for _ in range(num_centroids)]
    clustered_apis = [[] for _ in range(num_centroids)]
    for k, label in enumerate(labels):
        if len(clusters[label]) > max_cap:
            continue
        clusters[label].append(psis[k])
        clustered_apis[label].append(apis[k])
    return clusters, clustered_apis


def get_intra_cluster_jaccards(clustered_apis_k):

    dis_i = 0.
    count = 0.001
    for i, api_i in enumerate(clustered_apis_k):
        for j, api_j in enumerate(clustered_apis_k):
            if j <= i:
                continue
            dis_i += get_jaccard_distace(api_i, api_j)
            count += 1

    num_items = len(clustered_apis_k)
    return dis_i / count


def get_inter_cluster_jaccards(clustered_apis_j, clustered_apis_k):

    dis_ = 0.
    for api_i in clustered_apis_j:
        for api_j in clustered_apis_k:
            dis_ += get_jaccard_distace(api_i, api_j)

    num_items_1 = len(clustered_apis_j)
    num_items_2 = len(clustered_apis_k)

    return dis_ / (num_items_1 * num_items_2  + 0.0001)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_file', type=str, nargs=1,
                        help='input data file')
    parser.add_argument('--save', type=str, default='save',
                        help='directory to load model from')
    parser.add_argument('--top', type=int, default=10,
                        help='plot only the top-k labels')
    parser.add_argument('--index', required=True, choices=['b2', 'prog_psi_rev'])
    clargs = parser.parse_args()
    main(clargs)
