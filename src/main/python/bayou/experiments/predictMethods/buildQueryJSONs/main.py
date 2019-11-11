import random
import subprocess

import sys
import json
import os

from processJSON import processJSONs
from buildEmbeddings import embedding_server
from cleanup import cleanUp

from bayou.models.low_level_evidences.utils import read_config


'''

Experiment 1 :  only use sorrounding information

Experiment 2 :  sorrounding information + javadoc

Experiment 3 : use sorrounding information + javadoc + RT + FP

Experiment 4 : use sorrounding information + jD + RT + FP +
               10 pc of (apicalls, types, keywords)

Experiment 5 : use all of above + add sequence information
'''




def sampleFiles(queryFilesSampled, k):
    print("Sampling Files .... ", end="")
    sys.stdout.flush()
    # sample 10K random files
    randomFiles = random.sample(list(open('/home/ubuntu/github-java-files-train.txt', encoding = "ISO-8859-1").read().splitlines()) , k)

    with open(queryFilesSampled, "a") as f:
        for randFile in randomFiles:
            randFile = randFile[2:]
            randFile = '/home/ubuntu/java_projects/' + randFile + '\n'
            f.write(randFile)

    print("Done")
    return

def runDomDriver(queryFilesSampled, queryFilesInJson, logdir):
    print("Extracting Initial JSON ... ", end="")
    sys.stdout.flush()

    fileStdOut = logdir + '/L2stdoutDomDriver.txt'
    fileStdErr = logdir + '/L2stderrDomDriver.txt'
    JSONFiles =  logdir + '/JSONFiles'

    java_jar = "/home/ubuntu/bayou/tool_files/maven_3_3_9/batch_dom_driver/target/batch_dom_driver-1.0-jar-with-dependencies.jar"
    configFile = "/home/ubuntu/bayou/Java-prog-extract-config.json"

    with open(fileStdOut, "w") as f1 , open(fileStdErr, "w") as f2:
        subprocess.run(["java" , "-jar",  java_jar , queryFilesSampled, configFile ] , stdout=f1, stderr=f2)
        subprocess.run(["sed" , "-i",  "/^Going/d" , fileStdOut])

    with open(queryFilesInJson, 'w') as f:
        subprocess.run(["sed",  "s/.java$/.java.json/g", fileStdOut] , stdout=f)

    #subprocess.run(["while" , "read", "LINE;", "do", "cp", "$LINE", "JSONFiles/;", "done", "<", "L3JSONFiles.txt"], shell=True)


    print("Done")
    return




if __name__ == "__main__":

    logdir = "../log"
    queryFilesSampled = logdir + "/L1SampledQueryFileNamesfiles.txt"
    queryFilesInJson = logdir + '/L3JSONFiles.txt'

    #cleanUp(logdir = logdir)
    #sampleFiles(queryFilesSampled, k=100000)
    #runDomDriver(queryFilesSampled, queryFilesInJson, logdir)

    with open('/home/ubuntu/savedSearchModel/config.json') as f:
        config = read_config(json.load(f), chars_vocab=True)


    EmbS = embedding_server()
    for expNumber in range(11):
         exp_logdir = logdir + "/expNumber_" + str(expNumber)
         count = processJSONs(queryFilesInJson,  exp_logdir, config, expNumber = expNumber)
         EmbS.getEmbeddings(exp_logdir)
         print("Number of programs processed for exp " + str(expNumber) + " is "  + str(count))
