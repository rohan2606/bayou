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

        self.embedding_psi[index] = np.asarray( [ item for item in jsProgram['ev_psi'] ], dtype=np.float32)
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
       norm_a = tf.nn.l2_normalize(a,0)
       norm_b = tf.nn.l2_normalize(b,0)
       return 1 - tf.reduce_sum(tf.multiply(norm_a, norm_b))


    def measureDistance(self, embedding):

        self.distance = self.cosine_similarity(embedding, self.embedding_psi)
        return
