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

    JSONReader = parallelReadJSON('/home/ubuntu/DATABASE/', numThreads=numThreads, dimension=dimension, batch_size=batch_size, maxJSONs=70)
    listOfColDB = JSONReader.readAllJSONs()

    print ("Initiate Scanner")
    scanner = searchFromDB(listOfColDB, topK, batch_size)
    print ("Load Embedding")
    embIt = Embedding_iterator_WBatch('../log/EmbeddedProgramList.json', batch_size, dimension)
    print ("Searching Now!")

    k=0
    for embedding in embIt.embList:
        start = time.time()
        k = k+1
        topKProgs = scanner.searchAndTopKParallel(embedding, numThreads = numThreads, printProgs='no')
        end = time.time()
        print (k)
        print((end - start)/( batch_size))
