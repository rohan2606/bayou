import pickle
import json
import re

with open('dict_sequences_all_data.pkl', 'rb') as f:
    seq_dict = pickle.load(f)

print('Old Pickle Loaded')

dict_api_calls = dict()
count = 0
for key, value in seq_dict.items():
    value_key = set()
    all_sequences = value #seq_dict[key]
    all_sequences = eval(all_sequences.replace("u'", "'"))
    for seq in all_sequences:
        calls = seq['calls']
        for call in calls:
            call = re.sub('^\$.*\$', '', call)
            name = call.split('(')[0].split('.')[-1]
            name = name.split('<')[0]
            if name[0].islower(): # following Vijays convention in evidence.py but why?
                value_key.add(name)
    dict_api_calls[key] = list(value_key)
    count += 1

    if count % 10000 == 0:
        print('Done with ' + str(count)+ ' Programs')
        # break

with open('dict_api_all_data.pkl', 'wb') as fp:
    pickle.dump(dict_api_calls, fp, protocol=pickle.HIGHEST_PROTOCOL)



with open('dict_api_all_data.pkl', 'rb') as f:
    api = pickle.load(f)

