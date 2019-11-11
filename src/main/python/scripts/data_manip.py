import pickle
import numpy as np
import ijson 
import json

def load_data():
      print('Loading Data')
      with open('data/inputs.npy', 'rb') as f:
          inputs = pickle.load(f)
      # with open(, 'rb') as f:
      nodes = np.load('data/nodes.npy')
      edges = np.load('data/edges.npy')
      targets = np.load('data/targets.npy')


      js_programs = []
      with open('data/js_programs.json', 'rb') as f:
          for program in ijson.items(f, 'programs.item'):
              js_programs.append(program)
      return inputs, nodes, edges, targets, js_programs


def reduce_nums(inputs, nodes, edges, targets, js_programs):

    max_data = 1000000

    new_inputs = [inp[:max_data] for inp in inputs[:-1]]
    surr_new_inputs = [surr_inp[:max_data] for surr_inp in inputs[-1][:-1]]
    surr_fps_new_inputs = [surr_inp_fp[:max_data] for surr_inp_fp in inputs[-1][-1]]
    
    surr_new_inputs.append(surr_fps_new_inputs)
    new_inputs.append(surr_new_inputs)

    nodes = nodes[:max_data]
    edges = edges[:max_data]
    targets = targets[:max_data]
    js_programs = js_programs[:max_data]
    return new_inputs, nodes, edges, targets, js_programs




def save_new_data(inputs, nodes, edges, targets, js_programs):

    print('Saving...')
    with open('data_small/inputs.npy', 'wb') as f:
        pickle.dump(inputs, f, protocol=4) #pickle.HIGHEST_PROTOCOL)
    np.save('data_small/nodes', nodes)
    np.save('data_small/edges', edges)
    np.save('data_small/targets', targets)

    with open('data_small/js_programs.json', 'w') as f:
        json.dump({'programs': js_programs}, fp=f, indent=2)
    return


if __name__ == "__main__":
    i,n,e,t,j = load_data()
    i,n,e,t,j = reduce_nums(i,n,e,t,j)
    save_new_data(i,n,e,t,j)
