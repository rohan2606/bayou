from parallelReadJSON import parallelReadJSON
from searchFromDB import searchFromDB
from Embedding import Embedding_iterator_WBatch
from Embedding import Embedding
import time



if __name__=="__main__":


    numThreads = 30
    batch_size = 10
    dimension = 256
    topK = 10

    JSONReader = parallelReadJSON('/home/ubuntu/DATABASE/', numThreads=numThreads, dimension=dimension, batch_size=batch_size, maxJSONs=30)
    listOfColDB = JSONReader.readAllJSONs()

    print ("Initiate Scanner")
    scanner = searchFromDB(listOfColDB, topK, batch_size)
    print ("Load Embedding")
    embIt = Embedding_iterator_WBatch('../log/EmbeddedProgramList.json', batch_size, dimension)
    print ("Searching Now!")

    count, posCount = 0 , 0
    for embedding in embIt.embList:
        start = time.time()
        topKProgsBatch = scanner.searchAndTopKParallel(embedding, numThreads = numThreads, printProgs='no')

        for batch_id , topKProgs in enumerate(topKProgsBatch):
            # print ("=====================================================")

            desire = embedding.jsEmbedding[batch_id]
            # print (desire)

            for prog in topKProgs:
                # print ("---------------------------------------------------------")
                # print (prog.body)

                if prog.body == desire:
                    posCount += 1
                    break


        count = count + batch_size
        end = time.time()
        print ( str(posCount / count) +  " out of " + str(count) )
        print(  " Time Spent ::  " + str((end - start)/( batch_size)))
