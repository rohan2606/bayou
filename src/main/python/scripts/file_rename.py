import os


path = '/home/ubuntu/DATABASE/'

files = []
for r, d, f in os.walk(path):
    for file in f:
        if 'json' in file:
            files.append(os.path.join(r, file))

for f in files:
    fname = f.split('/')[-1]
    f_splits = fname.replace('.','_').split('_')

    file_num = f_splits[-2]
    new_file_name = '_'.join(f_splits[:-4] + [file_num]) + '.json'

    os.rename(f, new_file_name)
    print(f'Renamed {f} with {new_file_name}')
