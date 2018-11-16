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

from __future__ import print_function

# Use this script to split data between N processes. The script will accept a
# JSON file containing the data (such as DATA-testing.json, variational-*.json,
# etc.) and split it into N files

import sys
import simplejson
import ijson.backends.yajl2_cffi as ijson
import math
import argparse


def split(args):
    f = open(args.input_file[0], 'rb')
    assert(args.part>0 and args.part<100)
    start , end = (args.part-1)*args.step , args.part*args.step
    i = 0
    split_programs = []
    for program in ijson.items(f, 'programs.item'):
        print('Split part {} of size {} #Finished {} programs'.format(args.part, args.step, i), end='\r')
        if i == end:
            break
        if i < start:
            i += 1
            continue
        else:
            split_programs.append(program)
            i += 1

    print('')
    print("Writing to File")
    with open('{}-{:02d}.json'.format(args.input_file[0][:-5], args.part), 'w') as f:
        simplejson.dump({'programs': split_programs}, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, nargs=1,
                        help='input JSON file')
    parser.add_argument('--step', type=int, required=True,
                        help='step size to split JSON')
    parser.add_argument('--part', type=int, required=True,
                        help='current part')
    parser.add_argument('--python_recursion_limit', type=int, default=10000,
                        help='set recursion limit for the Python interpreter')

    args = parser.parse_args()
    sys.setrecursionlimit(args.python_recursion_limit)
    split(args)
