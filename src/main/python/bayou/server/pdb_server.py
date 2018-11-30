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
import logging.handlers
import os
from itertools import chain
from flask import request, Response, Flask
import subprocess



# called when a POST request is sent to the server at the index path
def _handle_http_post_request_index():

    request_json = request.data.decode("utf-8")  # read request string
    logging.debug("request_json:" + request_json)
    request_dict = json.loads(request_json)  # parse request as a JSON string

    request_type = request_dict['request type']

    if request_type == 'code search':
        codes = _handle_code_search_request(request_dict)
        return Response(asts, mimetype="application/json")
    elif request_type == 'shutdown':
        _shutdown()  # does not return

    return Response("")


# called when a GET request is sent to the server at the /pdbhealth path
def _handle_http_get_request_health():
    return Response("Ok")


# handle an asts generation request by generating asts
def _handle_code_search_request(request_dict, TopK=10):
    logging.debug("entering")

    programs = []
    program = {}
    program['a1'] = request_dict['a1'].item() # .item() converts a numpy element to a python element, one that is JSON serializable
    program['b1'] = [val.item() for val in evidence_b1]

    if 'TopK' in request_dict:
        TopK = program['TopK']

    programs.append(program)

    print('\nWriting to {}...'.format('/home/ubuntu/QueryProgWEncoding.json'), end='\n')

    with open('/home/ubuntu/QueryProgWEncoding.json', 'w') as f:
        json.dump({'programs': programs}, fp=f, indent=2)

    logging.debug(programs)

    PDB_Call_Script = "$PDB_HOME/bin/CodeSearch 8 localhost "
    PDB_Call_Script += str(parser.latent_size) + " "
    PDB_Call_Script += str(TopK) + " "
    PDB_Call_Script += "/home/ubuntu/QueryProgWEncoding.json" + " "
    PDB_Call_Script += "/home/ubuntu/TopPrograms.txt"

    subprocess.call([PDB_Call_Script])

    with open("/home/ubuntu/TopPrograms.txt") as f:
        js = json.load(f)

    logging.debug("exiting")
    return json.dumps(js, indent=2)



# terminates the Python process. Does not return.
def _shutdown():
    print("===================================")
    print("            PDB Stopping         ")
    print("===================================")
    os._exit(0)


def instatiatePDB():
    print("====================================================")
    print("    Loading Data Into PDB. Please Wait For Long.    ")
    print("====================================================")
    subprocess.call(["export PDB_HOME=/home/ubuntu/plinycompute"])
    subprocess.call(["export PDB_INSTALL=/tmp/pdb_install"])
    subprocess.call(["make -j 4 pdb_main"])
    subprocess.call(["$PDB_HOME/scripts/startCluster.sh standalone localhost"])
    subprocess.call(["ps aux | grep pdb"])
    PDB_Load_Script = "$PDB_HOME/bin/CodeSearchLoadData 8 localhost Y "
    PDB_Load_Script += str(parser.latent_size) + " "
    PDB_Load_Script += str(parser.data_dir)
    subprocess.call(["PDB_Load_Script"])
    return




if __name__ == '__main__':

    # Parse command line args.
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True, help='model directory to laod from')
    parser.add_argument('--logs_dir', type=str, required=False, help='the directories to store log information '
                                                                     'separated by the OS path separator')
    parser.add_argument('--data_dir', type=str, default='/home/ubuntu/DATABASE', help='the directory from where Program'
                                                                     'Data with a1/b1 to be retreived')
    parser.add_argument('--latent_size', type=int, default=128, help='latent size of your code search model')

    args = parser.parse_args()

    if args.logs_dir is None:
        dir_path = os.path.dirname(__file__)
        log_paths = [os.path.join(dir_path, "../../../logs/ast_server.log")]
    else:
        log_paths = [(d + "/pdb_server.log") for d in args.logs_dir.split(os.pathsep)]

    # ensure the parent directory of each log path exists or create it
    for log_path in log_paths:
        if not os.path.exists(os.path.dirname(log_path)):
            os.makedirs(os.path.dirname(log_path))

    # Create the logger for the application.
    logging.basicConfig(
        format='%(asctime)s,%(msecs)d %(levelname)-8s [%(threadName)s %(filename)s:%(lineno)d] %(message)s',
        datefmt='%d-%m-%Y:%H:%M:%S',
        level=logging.DEBUG,
        handlers=[logging.handlers.RotatingFileHandler(log_path, maxBytes=100000000, backupCount=9) for log_path in
                  log_paths])

    logging.debug("entering")  # can't move line up in program because logger not configured until this point

    # Set up HTTP server, but do no start it (yet).
    http_server = Flask(__name__)

    # Load model and start processing any sent requests.
    logging.info("loading data into PDB ")
    instatiatePDB(parser)
    logging.info("loaded data into PDB ")
    # route POST requests to / to _handle_http_post_request_index(...)
    http_server.add_url_rule("/", "index", lambda: _handle_http_post_request_index(), methods=['POST'])
    # route GET requests to /pdbhealth to _handle_http_get_request_health
    http_server.add_url_rule("/pdbhealth", "/pdbhealth", _handle_http_get_request_health, methods=['GET'])

    print("===================================")
    print("            PDB Ready            ")
    print("===================================")
    http_server.run(host='0.0.0.0', port=8081)  # does not return
    _shutdown()  # we don't shut down flask directly, but if for some reason it ever stops go ahead and stop Bayou
