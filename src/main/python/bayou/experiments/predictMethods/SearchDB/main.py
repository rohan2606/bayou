from parallelReadJSON import parallelReadJSON
from searchFromDB import searchFromDB
from Embedding import Embedding_iterator_WBatch
import time
import numpy as np

def rank_statistic(_rank, total, prev_hits, cutoff):
    cutoff = np.array(cutoff)
    hits = prev_hits + (_rank < cutoff)
    prctg = hits / total
    return hits, prctg

def ListToFormattedString(alist, Type):
    # Each item is right-adjusted, width=3
    if Type == 'float':
        formatted_list = ['{:.2f}' for item in alist]
        s = ','.join(formatted_list)
    elif Type == 'int':
        formatted_list = ['{:>3}' for item in alist]
        s = ','.join(formatted_list)
    return s.format(*alist)

if __name__=="__main__":

    numThreads = 20
    batch_size = 10
    maxJSONs = 70
    dimension = 256
    topK = 10000

    JSONReader = parallelReadJSON('/home/ubuntu/DATABASE/', numThreads=numThreads, dimension=dimension, batch_size=batch_size, maxJSONs=maxJSONs)
    listOfColDB = JSONReader.readAllJSONs()

    print ("Initiate Scanner")
    scanner = searchFromDB(listOfColDB, topK, batch_size)

    for expNumber in [0,1,2,3,4,5,6]:
	    print ("Load Embedding for ExpNumber :: "  +  str(expNumber) )
	    embIt = Embedding_iterator_WBatch('../log/expNumber_'+ str(expNumber) +'/EmbeddedProgramList.json', batch_size, dimension)
	    #print ("Searching Now!")

	    count = 0    
	    hit_points = [1,2,5,10,50,100,500,1000,5000,10000]
	    hit_counts = np.zeros(len(hit_points))


	    for kkk, embedding in enumerate(embIt.embList):
	        start = time.time()
	        scanner.addAColDB(embedding.js, dimension, batch_size)
	        topKProgsBatch = scanner.searchAndTopKParallel(embedding, numThreads = numThreads, printProgs='no')

		 
	        for batch_id , topKProgs in enumerate(topKProgsBatch):
	            desire = embedding.js[batch_id]['body']
	            desireAPIcalls = embedding.js[batch_id]['testapicalls']
	            rank = topK + 1

	            for j, prog in enumerate(topKProgs):
                        flag=True
                        for api in desireAPIcalls:
                            if api not in prog.body:
                                flag=False
                        if flag == True:
                            rank = j
                            break
                        '''
                        if desire in prog.body :
                            rank = j
                            break
                        '''
	            count += 1
	            hit_counts, prctg = rank_statistic(rank, count, hit_counts, hit_points)
		     
	        scanner.deleteLastColDB()

	        end = time.time()
	        print('Searched {} Hit_Points {} :: Percentage Hits {}'.format
                          (count, ListToFormattedString(hit_points, Type='int'), ListToFormattedString(prctg, Type='float')))
	        if kkk % 9 == 0 and kkk > 0:
	              #print('Searched {} Hit_Points {} :: Percentage Hits {}'.format
		      #	  (count, ListToFormattedString(hit_points, Type='int'), ListToFormattedString(prctg, Type='float')))
	              break
		#print(  " Time Spent ::  " + str((end - start)/( batch_size)))
