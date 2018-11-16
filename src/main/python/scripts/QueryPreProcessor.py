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
import math
import argparse


#INPUT
#
# import java.io.*;
# import java.util.*;
# public class TestIO {
#      void read(File file) {
#          {
#              /// call:readLine type:FileReader type:BufferedReader
#          }
#      }
# }

# OUTPUT
# import java.io.*;
# import java.util.*;
# public class TestIO {
#
#      /**
#
#      call:readLine
#      type:FileReader
#      type:BufferedReader
#
#      */
#      public void read(File file) {
#          {
#              __PDB_FILL__();
#          }
#      }
# }

def PreProcess(args):
    f = open(args.input_file[0], 'r')
    content = f.readlines()

    apicalls = []
    types = []
    keywords = []
    outLines = []
    for line in content:
        if not line.strip().startswith('///'):
            outLines.append(line)
        else:
            line = line.strip().replace("/","")
            strings = line.split(" ")
            for string in strings:
                if 'call:' in string:
                    apicalls.append(string)
                if 'type:' in string:
                    types.append(string)
                if 'keyword:' in string:
                    keywords.append(string)

            indent = " " * (len(outLines[-1]) - len(outLines[-1].lstrip(' ')))
            outLines.insert(len(outLines)-1, indent + '/**' + "\n".join([indent + string for string in strings])  + '\n' + indent + '*/\n')
            indent = indent + " " * 5
            outLines.append(indent+"__PDB_FILL__();\n")

    # with open(args.input_file[0][:-5] + "Parsed.java", 'w') as target:
    with open('/home/ubuntu/ParsedJava.java', 'w') as target:
        target.writelines(outLines)
    # print('')
    # print("Writing to File")
    # with open('{}-{:02d}.json'.format(args.input_file[0][:-5], args.part), 'w') as f:
    #     simplejson.dump({'programs': split_programs}, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, nargs=1,
                        help='input JSON file')

    args = parser.parse_args()
    PreProcess(args)
