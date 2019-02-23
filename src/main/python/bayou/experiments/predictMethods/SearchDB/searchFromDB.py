from multiprocessing.dummy import Pool as ThreadPool
#from multiprocessing import Pool
from Embedding import Embedding




class searchFromDB():

    def __init__(self, listOfColDB, topK, batch_size):
        self.listOfColDB = listOfColDB
        self.topK = topK
        self.batch_size = batch_size



    def searchAndTopK(self, colDBs):

        # searchEmbedding = Embedding()

        # print ("TopK in a thread")
        progList = [[] for i in range(self.batch_size)]
        for colDB in colDBs:
            colDB.measureDistance(  self.searchEmbedding )

            topKList = colDB.topKProgs( self.topK )
            for batch_id , topK in enumerate(topKList):
                progList[batch_id].extend( topK )
        # print ("TopK in a thread finished")
        return progList


    def searchAndTopKParallel(self, searchEmbedding,  numThreads = 32, printProgs='no'):

        print("Starting parallel search")

        # self.programsDB.resetDataBaseEmbeddings()
        self.searchEmbedding = searchEmbedding
        colDBChunks = [self.listOfColDB[i::numThreads] for i in range(numThreads)]

        pool = ThreadPool(processes=numThreads)
        print("Starting to Pool")
        threadTopKProgs = pool.map(self.searchAndTopK, colDBChunks)
        pool.close()
        pool.join()
        print("Done with Pooling")
        #flatten them

        topProgramForBatch=[[] for i in range(self.batch_size)]
        for t, topProgramsForThread_t in enumerate(threadTopKProgs):
            for j, topProgsForBatch_j in enumerate(topProgramsForThread_t):
                topProgramForBatch[j].extend(topProgsForBatch_j)


        opTopProgramForBatch = []
        for j, topKProgs in enumerate(topProgramForBatch):
            [prog.setDistance(j) for prog in topKProgs]
            temp = sorted(topKProgs, key=lambda x: x.getDistance(j), reverse=True)[:self.topK]
            opTopProgramForBatch.append(temp)
            # [print (prog.distance[j]) for prog in topKProgs]


        return opTopProgramForBatch

    # def printTopProgs(self, topKBatchProgs):
    #
    #     for i, topKProgs in enumerate(topKBatchProgs):
    #         print (self.searchEmbedding.jsEmbedding[i]['body'])
    #         for j, topKProgs in enumerate(topKBatchProgs):
    #             for rank, program in enumerate(topKProgs):
    #                 program.print_self(rank)
