from parallelReadJSON import parallelReadJSON
from searchFromDB import searchFromDB
from Embedding import Embedding_iterator_WBatch
from utils import rank_statistic, ListToFormattedString

import time
import numpy as np
import re
import json

logdir = "../log"


if __name__=="__main__":

    numThreads = 32
    batch_size = 5
    minJSONs = 0
    maxJSONs = 69 #230
    dimension = 256
    topK = 10000



    JSONReader = parallelReadJSON('/home/ubuntu/DATABASE/', numThreads=numThreads, dimension=dimension, batch_size=batch_size, minJSONs=minJSONs, maxJSONs=maxJSONs)
    listOfColDB = JSONReader.getSearchDatabase()


    print ("Initiate Scanner")
    scanner = searchFromDB(listOfColDB, topK, batch_size)

    for expNumber in range(7):
        print ("Load Embedding for ExpNumber :: "  +  str(expNumber) )
        embIt = Embedding_iterator_WBatch('../log/expNumber_'+ str(expNumber) +'/EmbeddedProgramList.json', batch_size, dimension)

        count = 0
        hit_points = [1,2,5,10,50,100,500,1000,5000,10000]
        hit_counts = np.zeros(len(hit_points))

        JSONList = []
        for kkk, embedding in enumerate(embIt.embList):
            scanner.addAColDB(embedding.js, dimension, batch_size)
            topKProgsBatch = scanner.searchAndTopKParallel(embedding, numThreads = numThreads, printProgs='no')


            for batch_id , topKProgs in enumerate(topKProgsBatch):

                topProgramDict = dict()

                desire = embedding.js[batch_id]['body']
                desire_apis = embedding.js[batch_id]['testapicalls']

                topProgramDict['desiredProg'] = desire
                rank = topK + 1

                programList = list()

                for j, prog in enumerate(topKProgs):
                    if j < 10:
                        programList.append({j:prog.body})
                    
                    jaccardMatch = True
                    for desire in desire_apis :
                        if desire not in prog.body:
                           jaccardMatch = False
                           break
                    
                    if jaccardMatch == True and j < rank:
                        rank = j

                topProgramDict['topPrograms'] = programList
                count += 1
                hit_counts, prctg = rank_statistic(rank, count, hit_counts, hit_points)

                JSONList.append(topProgramDict)

            scanner.deleteLastColDB()
            print('Searched {} Hit_Points {} :: Percentage Hits {}'.format
                          (count, ListToFormattedString(hit_points, Type='int'), ListToFormattedString(prctg, Type='float')))
            if kkk % 9 == 0 and kkk > 0:
                with open(logdir + "/expNumber_" + str(expNumber) + '/L5TopProgramList.json', 'w') as f:
                     json.dump({'topPrograms': JSONList}, fp=f, indent=2)
                break
