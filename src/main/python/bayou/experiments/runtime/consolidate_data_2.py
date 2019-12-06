import json
import os
import matplotlib.pyplot as plt
import numpy as np


from functools import reduce
from itertools import chain
import bayou.models.low_level_evidences.evidence
from bayou.models.low_level_evidences.utils import gather_calls
import scripts.ast_extractor
from bayou.experiments.predictMethods.SearchDB.utils import get_jaccard_distace_api


def read_json(fname):
    with open(fname) as f:
        js = json.load(f)
    return js


# This method is not to be used
# as all other info is available
# in the other log files
def get_golden_progs(fname):
    js = read_json(fname)
    time = js["Time"]
    programs = js["Programs"]
    return time, programs


def get_apicalls(program_ast):
   calls = gather_calls(program_ast['_nodes'])
   apicalls = list(set(chain.from_iterable([bayou.models.low_level_evidences.evidence.APICalls.from_call(call)
                                            for call in calls])))
   return apicalls


def get_log_iter_progs(fname, golden_progs):
    js = read_json(fname)
    iter_num = js["Iteration"]
    #time = js["Time"]
    #programs = js["Programs"]
    slowdown = js["Slowdown"]
    deviations = [float(item) for item in js["Deviations"]]
    variations = [float(item) for item in js["Variations"]]

    jaccard = 0.
    count = 0
    for program1 in js['Programs'][:10]:
        ast1 = program1['ast']['prog_ast']
        apis1 = get_apicalls(ast1)
        for program2 in golden_progs[:10]:
            ast2 = program2['ast']['prog_ast']
            apis2 = get_apicalls(ast2)
            jaccard += get_jaccard_distace_api(apis1, apis2)
            count += 1
    #exist_distance = js["Distances"]["AST Exist[1/3/5/10/100]"]
    #jac_distance = js["Distances"]["AST Jaccard[1/3/5/10/100]"]
    return slowdown, variations, jaccard/count


def multi_log_data(log_file='log', max_log_data=25):
    prefix = log_file + '/mc_iter_logger_'
    suffix = '.json'

    golden_time, golden_progs = get_golden_progs(log_file + '/golden_prog_logger_cpu.json')

    slowdowns = []
    exist_distances = []
    jac_distances = []

    # variations need not be accumulated, already done in the log file
    for file_id in range(0, max_log_data):
        file_name = prefix + str(file_id) + suffix
        slowdown, variations, jac_distance = get_log_iter_progs(file_name, golden_progs)
        slowdowns.append(float(slowdown))
        jac_distances.append(jac_distance)
    return slowdowns, variations, jac_distances



def transpose_distance(distances_with_iters):

    distance10 = [float(distance_iter) for distance_iter in distances_with_iters]
    
    return [distance10]




def get_many_log_data():
    folders = get_log_folders()
    
    arr_of_slowdowns = []
    arr_of_variations = []
    arr_of_exist_distances = []
    arr_of_jac_distances = []

    #accepted_folders = ['log2_Br_rL', 'log3_arraylist_add2List', 'log_hashmap', 'log_split_ok_jaccard', 'log_int2List']
    #accepted_folders = ['log2_Br_rL', 'log3_arraylist_add2List', 'log_hashmap', 'log_int2List']
    accepted_folders = ['log1_Br_rL', 'log3_arraylist_add2List', 'log_hashmap', 'log_int2List']
    accepted_folders = ['log_hashmap']
    for folder in folders:
        if not folder in accepted_folders:
           continue
        slowdowns, variations, jac_distances = get_major_log_data(folder)
        arr_of_slowdowns.append(slowdowns)
        arr_of_variations.append(variations)
        print(jac_distances)
        arr_of_jac_distances.append(jac_distances)
    
    arr_of_slowdowns = np.array(arr_of_slowdowns)
    arr_of_variations = np.array(arr_of_variations)
    arr_of_jac_distances = np.array(arr_of_jac_distances)

    
    arr_of_slowdowns = np.mean(arr_of_slowdowns, axis=0)
    arr_of_variations = np.mean(arr_of_variations, axis=0)
    arr_of_jac_distances = np.mean(arr_of_jac_distances, axis=0)
    
    return arr_of_slowdowns, arr_of_variations, arr_of_jac_distances
    
    

def get_major_log_data(log_file):
    slowdowns, variations, jac_distances = multi_log_data(log_file=log_file , max_log_data=30)
    jac_distances = transpose_distance(jac_distances)
    return slowdowns, variations, jac_distances


def get_log_folders():
    output = [name for name in os.listdir('./') if 'log' in name]
    return output


def plot(slowdowns, variations,  jaccard_distance, name='output.png'):
    fig = plt.figure()    
    ax = plt.subplot(111)      
    width=0.25
  
    slowdowns = np.insert(slowdowns, 1, slowdowns[0]) 
    variations = np.insert(variations, 0, variations[0])
    jaccard_distance = np.insert(jaccard_distance, 0, jaccard_distance[0])


 
    jaccard_distance = [(1 - val) for val in jaccard_distance] 
    #plt.ylim(0.0, 65.0)

    freq = 5
    jaccard_distance = jaccard_distance[::freq]
    slowdowns = slowdowns[::freq]
    variations = variations[::freq]
    N = len(slowdowns)
    ind = np.arange(N) 
        
    # plt bar supports yerr arg, test that
    #plt.bar(ind, exist_distance, width, color='b', label='Avg. Exist Distance')
    #plt.bar(ind+width, jaccard_distance, width, color='r', label='Avg. Jaccard Distance')
    #plt.bar(ind+2*width, variations, width, color='g', label='Avg. Co-eff of Var')
    
    plt.plot(ind, jaccard_distance, color='darkgreen', marker='v', linewidth=2, markersize=6, label='Avg. Jaccard Distance')
    plt.plot(ind, variations, marker='X',color='blueviolet', linewidth=2, markersize=6, label='Avg. Co-eff of Var')
    
    plt.ylabel('Distance')      
    plt.xlabel('Iteration')
       
    plt.grid(True)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
          fancybox=True, shadow=True, ncol=3)

    #plt.legend() 
    axes2 = plt.twinx()
    #plt.ylim(0.0, 65.0)
    axes2.bar(ind, slowdowns, width, color='coral', edgecolor='k', alpha=1.0, linewidth=1.8) #, yerr=menStd, label='Men means')
    #plt.bar(ind+width, womenMeans, width, color='y', label='Women means')
    axes2.plot(ind, slowdowns, color='k', linestyle='dashed', label='Slowdown')
    #axes2.set_ylim(0, max(y))
    axes2.set_ylabel('Slowdown')
    #axes2.set_xlabel('Iterations')
    box = axes2.get_position()
    axes2.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

    plt.xticks(ind, [i*freq for i in ind] )    
    #axes2.xaxis.grid(True)
    #axes2.yaxis.grid(True)

    ax.set_zorder(axes2.get_zorder()+1)
    ax.patch.set_visible(False)
    plt.savefig(name)




if __name__ == '__main__':
    slowdowns, variations, jac_distances = get_many_log_data()

    plot(slowdowns, variations, jac_distances, name='output_distance_10.png')








