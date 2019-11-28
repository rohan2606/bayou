import numpy as np



class MyColumnDatabaseWBatch():

    def __init__(self, numItems, dimension, batch_size):
        self.numItems = numItems
        self.dimension = dimension
        self.batch_size = batch_size

        self.embedding_psi = np.zeros([numItems, dimension], dtype=np.float32)

        self.programs = [None for i in range(self.numItems)]
        self.distance = np.full([ self.batch_size, self.numItems ], np.inf, dtype=np.float32)



    def setValues(self, jsProgram, decProg, index):

        self.embedding_psi[index] = np.asarray( [ item for item in jsProgram['prog_psi_rev'] ], dtype=np.float32)
        self.programs[index] = decProg
        return


    def topKProgs(self, k=10):

        topKforBatch = []
        modK = min(k, self.numItems)
        for item in range(self.batch_size):
            topKids = (-self.distance[item]).argsort()[:modK] # slow to get topK
            topKProgs = [(self.programs[_id], self.distance[item][_id] ) for _id in topKids]
            topKforBatch.append(topKProgs)

        return topKforBatch


    def cosine_similarity(self, a, b):
       norm_denom_a = np.linalg.norm(a,axis=1)
       norm_a = a/(norm_denom_a[:,None]+0.0001)
       
       norm_denom_b = np.linalg.norm(b, axis=1)
       norm_b = b/(norm_denom_b[:, None]+0.0001)
       
       sim = np.dot(norm_a, np.transpose(norm_b))
       return sim #(1 - sim)


    def measureDistance(self, embedding):

        self.distance = self.cosine_similarity(embedding.psi, self.embedding_psi)
        return
