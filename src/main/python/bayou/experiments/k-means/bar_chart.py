import numpy as np
import matplotlib.pyplot as plt
import json
# data to plot

with open('prob_model.json') as pf:
     js_prob = json.load(pf)['jaccard_intra_cluster']

with open('non_prob_model.json') as pf:
     js_non_prob = json.load(pf)['jaccard_intra_cluster']

n_groups = len(js_prob)



fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, js_prob, bar_width,
alpha=opacity,
color='b',
label='Bayesian')

rects2 = plt.bar(index + bar_width, js_non_prob, bar_width,
alpha=opacity,
color='g',
label='Non-Probabilistic')

plt.xlabel('Cluster Number')
plt.ylabel('Average Jaccard Similarity of APIs')
#plt.title('Similarity of Programs in Embedding Space')
plt.xticks(index + bar_width, [x+1 for x in range(n_groups)])
plt.legend()

plt.grid(which='major', linestyle='-', linewidth='0.05', color='k')

#plt.grid()
plt.tight_layout()
plt.savefig('comparison.png')
