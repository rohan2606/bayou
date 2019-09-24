import numpy as np



class MyColumnDatabaseWBatch():

    def __init__(self, numItems, dimension, batch_size):
        self.numItems = numItems
        self.dimension = dimension
        self.batch_size = batch_size

        self.numpy_A = np.zeros([numItems], dtype=np.float32)
        self.numpy_B = np.zeros([numItems, dimension], dtype=np.float32)
        self.numpy_ProbY =  np.zeros([numItems], dtype=np.float32)

        self.programs = [None for i in range(self.numItems)]
        self.distance = np.full([ self.batch_size, self.numItems ], np.inf, dtype=np.float32)



    def setValues(self, jsProgram, decProg, index):

        self.numpy_ProbY[index] = np.asarray(jsProgram['ProbY'] , dtype=np.float32)
        self.numpy_B[index] = np.asarray( [ item for item in jsProgram['b2'] ], dtype=np.float32)
        self.numpy_A[index] = np.asarray(jsProgram['a2'], dtype=np.float32)
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




    def measureDistance(self, embedding):

        latent_size = np.shape(embedding.B)[1]

        a_star =  embedding.A[:,None] + self.numpy_A[None, :]  + 0.5 # shape is [batch_size , numItems]
        b_star = embedding.B[:,None,:] + self.numpy_B[None, :, :]   # shape is [batch_size, numItems, latent_size]

        ab1 = np.sum( (np.square(embedding.B) / (4*embedding.A[:,None]) ), axis=1) + 0.5 * latent_size * np.log(-embedding.A / np.pi) # shape is (batch_size)
        ab2 = np.sum( (np.square(self.numpy_B) / (4*self.numpy_A[:, None]) ), axis=1) + 0.5 * latent_size * np.log(-self.numpy_A/np.pi) # shape is [numItems]
        ab_star = np.sum((np.square(b_star) /  (4 * a_star[:,:,None])), axis=2) + 0.5 *  latent_size * np.log(-a_star/np.pi) # shape is [batch_size , numItems]

        cons = 0.5 * latent_size * np.log( 2*np.pi )
        self.distance = ab1[:,None] + ab2[None,:] - ab_star - cons #+ self.numpy_ProbY[None,:]
        return
