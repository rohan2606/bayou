import json #ijson.backends.yajl2_cffi as ijson
import sys
import os

PROGS_PER_FILE=100000
MIN_JSONs = int(sys.argv[1])
MAX_JSONs = int(sys.argv[2])
buffer_ = []

op_file_id = MIN_JSONs

for file_id in range(MIN_JSONs,MAX_JSONs):
    fileName = 'Program_output_' + str(file_id) + '.json'
    print("Working on file :: " + fileName)
    with open( fileName , 'r') as f:
        js = json.load(f)

    programs = js['programs']
    buffer_.extend(programs)

    if len(buffer_) >= PROGS_PER_FILE:
       op_file_name = 'Codec_data_' + str(op_file_id) + '.json'
       with open(op_file_name, 'w') as f:
            json.dump({"programs":buffer_[:PROGS_PER_FILE]}, f, indent=4)
       buffer_ = buffer_[PROGS_PER_FILE:]
       op_file_id += 1
       print("Wrote on file :: " + op_file_name)

    os.remove(fileName)
    print("Removed :: " + fileName)


if len(buffer_) >= 0:
   op_file_name = 'Codec_data_partial_' + str(op_file_id) + '.json'
   with open(op_file_name, 'w') as f:
        json.dump({"programs":buffer_}, f, indent=2)
   print("Wrote on file :: " + op_file_name)

