from parallelReadJSON import parallelReadJSON
from searchFromDB import searchFromDB
from Embedding import Embedding_iterator_WBatch
import numpy as np
import simplejson as json
from utils import *

if __name__=="__main__":

    numThreads = 30
    batch_size = 5
    minJSONs = 0
    maxJSONs = 3 #230
    dimension = 256
    topK = 10000

    JSONReader = parallelReadJSON('/home/ubuntu/DATABASE/', numThreads=numThreads, dimension=dimension, batch_size=batch_size, minJSONs=minJSONs , maxJSONs=maxJSONs)
    listOfColDB = JSONReader.readAllJSONs()

    # dicts coming from outside
    dict_api_calls = get_api_dict()
    dict_ast = get_ast_dict()


    print ("Initiate Scanner")
    scanner = searchFromDB(listOfColDB, topK, batch_size)
    hit_points = [1,2,5,10,50,100,500,1000,5000,10000]

    for expNumber in [0,1,2,3,4,5,6]:
        print ("Load Embedding for ExpNumber :: "  +  str(expNumber) )
        embIt = Embedding_iterator_WBatch('../log/expNumber_'+ str(expNumber) +'/EmbeddedProgramList.json', batch_size, dimension)

        # for first hit
        hit_counts_first_hit_total = np.zeros(len(hit_points))
        count = 0

        # for precision
        hit_count_jaccard_api_total = np.zeros(len(hit_points))
        hit_counts_exact_match_total = np.zeros(len(hit_points))
        hit_counts_ast_total = np.zeros(len(hit_points))
        hit_counts_seq_total = np.zeros(len(hit_points))

        # for storage
        JSONList=[]

        for kkk, embedding in enumerate(embIt.embList):
            topKProgsBatch = scanner.searchAndTopKParallel(embedding, numThreads = numThreads)

            for batch_id , topKProgs in enumerate(topKProgsBatch):
                desiredBody, desireAPIcalls, desireAST = get_your_desires( embedding.js[batch_id] )

                # for first hit
                firstHitRank = topK + 1

                # for precision
                hitPtId = 0
                hit_counts_jaccard_api = np.zeros(len(hit_points))
                hit_counts_exact_match = np.zeros(len(hit_points))
                hit_counts_ast = np.zeros(len(hit_points))
                hit_counts_seq = np.zeros(len(hit_points))

                # for storage
                topProgramDict = embedding.js[batch_id]
                del topProgramDict['a2'],topProgramDict['b2'],topProgramDict['ProbY'], topProgramDict['ast'], topProgramDict['body'], topProgramDict['testapicalls']
                topProgramDict['desiredProg'] = desiredBody
                topProgramDict['desireAPIcalls'] = desireAPIcalls
                topProgramDict['desireAST'] = desireAST
                programList = list()

                for j, prog in enumerate(topKProgs):

                    # for first hit
                    if desiredBody in prog.body and j < firstHitRank:
                        firstHitRank = j



                    # for precision
                    key = prog.fileName + "/" + prog.methodName
                    hit_counts_jaccard_api[hitPtId] += int(jaccardSimAPI( dict_api_calls[key] , desireAPIcalls ))
                    hit_counts_exact_match[hitPtId] += int(exact_match( prog.body , desiredBody ))
                    hit_counts_ast[hitPtId] += int(exact_match_ast( dict_ast[key] , desireAST ))
                    hit_counts_seq[hitPtId] += 1 #int(exact_match_ast( dict_ast[key] , desireAST ))

                    ## For precision calculations
                    if (j+1)  == hit_points[hitPtId]:
                        hitPtId += 1
                        if (hitPtId <10):
                            hit_counts_jaccard_api[hitPtId] += hit_counts_jaccard_api[hitPtId-1]
                            hit_counts_exact_match[hitPtId] += hit_counts_exact_match[hitPtId-1]
                            hit_counts_ast[hitPtId] += hit_counts_ast[hitPtId-1]
                            hit_counts_seq[hitPtId] += hit_counts_seq[hitPtId-1]

                            ## For storage
                    if j < 100:
                        # programList.append({j:prog.body})
                        programList.append({j: dict_ast[key].replace("u'", "'") })

                #for first hit
                count += 1
                hit_counts_first_hit_total, prctg_first_hit = rank_statistic(firstHitRank, count, hit_counts_first_hit_total, hit_points)

                # for precision
                hit_count_jaccard_api_total += hit_counts_jaccard_api
                hit_counts_exact_match_total += hit_counts_exact_match
                hit_counts_ast_total += hit_counts_ast
                hit_counts_seq_total += hit_counts_seq


                # for storage
                topProgramDict['first_hit_rank'] = firstHitRank
                topProgramDict['topPrograms'] = programList
                JSONList.append(topProgramDict)

            # for precision
            prctg_jaccard_api = np.zeros_like(hit_count_jaccard_api_total)
            prctg_exact_match = np.zeros_like(hit_counts_exact_match_total)
            prctg_ast = np.zeros_like(hit_counts_ast_total)
            prctg_seq = np.zeros_like(hit_counts_seq_total)
            for i in range(len(hit_points)):
                denom = float(hit_points[i] * batch_size * (kkk+1))
                prctg_jaccard_api[i] = hit_count_jaccard_api_total[i] / denom
                prctg_exact_match[i] = hit_counts_exact_match_total[i] / denom
                prctg_ast[i] = hit_counts_ast_total[i] / denom
                prctg_seq[i] = hit_counts_seq_total[i] / denom

            print('Srchd {} Hit_Pts {} :: Cumulv First Hit {}'.format
            (batch_size * (kkk+1), ListToFormattedString(hit_points, Type='int'), ListToFormattedString(prctg_first_hit, Type='float')))
            print('Srchd {} Hit_Pts {} :: Prec API Jaccard {}'.format
            (batch_size * (kkk+1), ListToFormattedString(hit_points, Type='int'), ListToFormattedString(prctg_jaccard_api, Type='float')))
            print('Srchd {} Hit_Pts {} :: Prec Exact Match {}'.format
            (batch_size * (kkk+1), ListToFormattedString(hit_points, Type='int'), ListToFormattedString(prctg_exact_match, Type='float')))
            print('Srchd {} Hit_Pts {} :: Prec  AST  Match {}'.format
            (batch_size * (kkk+1), ListToFormattedString(hit_points, Type='int'), ListToFormattedString(prctg_ast, Type='float')))
            print('Srchd {} Hit_Pts {} :: Prec  Seq  Match {}'.format
            (batch_size * (kkk+1), ListToFormattedString(hit_points, Type='int'), ListToFormattedString(prctg_seq, Type='float')))
            print()

            if kkk % 9 == 0 and kkk > 0:
                with open("../log" + "/expNumber_" + str(expNumber) + '/L5TopProgramList.json', 'w') as f:
                    json.dump({'topPrograms': JSONList}, fp=f, indent=2)
                break
