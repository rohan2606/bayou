import json
import os
import matplotlib.pyplot as plt
import numpy as np

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


def get_log_iter_progs(fname):
    js = read_json(fname)
    iter_num = js["Iteration"]
    #time = js["Time"]
    #programs = js["Programs"]
    slowdown = js["Slowdown"]
    deviations = [float(item) for item in js["Deviations"]]
    variations = [float(item) for item in js["Variations"]]
    exist_distance = js["Distances"]["AST Exist[1/3/5/10/100]"]
    jac_distance = js["Distances"]["AST Jaccard[1/3/5/10/100]"]
    return slowdown, variations, exist_distance, jac_distance


def multi_log_data(log_file='log', max_log_data=25):
    prefix = log_file + '/mc_iter_logger_'
    suffix = '.json'

    slowdowns = []
    exist_distances = []
    jac_distances = []

    # variations need not be accumulated, already done in the log file
    for file_id in range(0, max_log_data):
        file_name = prefix + str(file_id) + suffix
        slowdown, variations, exist_distance, jac_distance = get_log_iter_progs(file_name)
        slowdowns.append(float(slowdown))
        exist_distances.append(exist_distance)
        jac_distances.append(jac_distance)
    return slowdowns, variations, exist_distances, jac_distances



def transpose_distance(distances_with_iters):

    distance1 = [float(distance_iter[0]) for distance_iter in distances_with_iters]
    distance3 = [float(distance_iter[1]) for distance_iter in distances_with_iters]
    distance5 = [float(distance_iter[2]) for distance_iter in distances_with_iters]
    distance10 = [float(distance_iter[3]) for distance_iter in distances_with_iters]
    distance100 = [float(distance_iter[4]) for distance_iter in distances_with_iters]
    
    return [distance1, distance3, distance5, distance10, distance100]




def get_many_log_data():
    folders = get_log_folders()
    
    arr_of_slowdowns = []
    arr_of_variations = []
    arr_of_exist_distances = []
    arr_of_jac_distances = []

    #accepted_folders = ['log2_Br_rL', 'log3_arraylist_add2List', 'log_hashmap', 'log_split_ok_jaccard', 'log_int2List']
    accepted_folders = ['log2_Br_rL', 'log3_arraylist_add2List', 'log_hashmap', 'log_int2List']
    for folder in folders:
        if not folder in accepted_folders:
           continue
        slowdowns, variations, exist_distances, jac_distances = get_major_log_data(folder)
        arr_of_slowdowns.append(slowdowns)
        arr_of_variations.append(variations)
        arr_of_exist_distances.append(exist_distances)
        arr_of_jac_distances.append(jac_distances)
    
    arr_of_slowdowns = np.array(arr_of_slowdowns)
    arr_of_variations = np.array(arr_of_variations)
    arr_of_exist_distances = np.array(arr_of_exist_distances)
    arr_of_jac_distances = np.array(arr_of_jac_distances)

    
    arr_of_slowdowns = np.mean(arr_of_slowdowns, axis=0)
    arr_of_variations = np.mean(arr_of_variations, axis=0)
    arr_of_exist_distances = np.mean(arr_of_exist_distances, axis=0)
    arr_of_jac_distances = np.mean(arr_of_jac_distances, axis=0)
    return arr_of_slowdowns, arr_of_variations, arr_of_exist_distances, arr_of_jac_distances
    
    

def get_major_log_data(log_file):
    slowdowns, variations, exist_distances, jac_distances = multi_log_data(log_file=log_file , max_log_data=30)
    exist_distances = transpose_distance(exist_distances)
    jac_distances = transpose_distance(jac_distances)
    return slowdowns, variations, exist_distances, jac_distances


def get_log_folders():
    output = [name for name in os.listdir('./') if 'log' in name]
    return output


def plot(slowdowns, variations, exist_distance, jaccard_distance, name='output.png'):
    plt.figure()          
    N = len(slowdowns)
    ind = np.arange(N)
    
    exist_distance = [(1-val) for val in exist_distance] 
    jaccard_distance = [(1 - val) for val in jaccard_distance] 
    #plt.ylim(0.0, 65.0)
    #plt.bar(ind, menMeans, width, color='r', yerr=menStd, label='Men means')
    #plt.bar(ind+width, womenMeans, width, color='y', yerr=womenStd, label='Women means')
    plt.plot(ind, exist_distance, color='r', label='Exist Distance')
    plt.plot(ind, jaccard_distance, color='b', label='Jaccard Distance')
    plt.plot(ind, variations, color='k', label='Co-efficient of variation')
    plt.ylabel('Distances')      
    
    y = variations
    
    axes2 = plt.twinx()
    #plt.ylim(0.0, 65.0)
    width=0.35
    #axes2.bar(ind, slowdowns, width, color='r') #, yerr=menStd, label='Men means')
    #plt.bar(ind+width, womenMeans, width, color='y', yerr=womenStd, label='Women means')
    axes2.plot(ind, slowdowns, color='k', label='Slowdown')
    #axes2.set_ylim(0, max(y))
    axes2.set_ylabel('Slowdowns')
    
    plt.savefig(name)



if __name__ == '__main__':
    slowdowns, variations, exist_distances, jac_distances = get_many_log_data()

    plot(slowdowns, variations, exist_distances[0], jac_distances[0], name='output_distance_1.png')
    plot(slowdowns, variations, exist_distances[1], jac_distances[1], name='output_distance_3.png')
    plot(slowdowns, variations, exist_distances[2], jac_distances[2], name='output_distance_5.png')
    plot(slowdowns, variations, exist_distances[3], jac_distances[3], name='output_distance_10.png')
    plot(slowdowns, variations, exist_distances[4], jac_distances[4], name='output_distance_100.png')








