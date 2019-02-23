# from parallelReadJSON import parallelReadJSON

# class MyDataBase():
#
#     def MyDataBase(self):
#         JSONReader = parallelReadJSON()
#         self.Database = JSONReader.readAllJSONs(numThreads=8)
#
#
#     def resetDataBaseEmbeddings(self):
#         for item in self.Database:
#             item.reset_distance()
#
#
#     def topKDataBase(self):
#         return
#
#
#


import numpy as np
#
# class MyColumnDatabase():
#
#     def __init__(self, numItems, dimension):
#         self.numItems = numItems
#         self.dimension = dimension
#
#         self.numpy_A = np.zeros([numItems], dtype=np.float32)
#         self.numpy_B = np.zeros([numItems, dimension], dtype=np.float32)
#         self.numpy_ProbY =  np.zeros([numItems], dtype=np.float32)
#
#         self.programs = [None for i in range(numItems)]
#         self.distance = np.full([numItems], np.inf, dtype=np.float32)
#
#
#
#     def setValues(self, jsProgram, decProg, index):
#
#         self.numpy_ProbY[index] = np.asarray(jsProgram['ProbY'] , dtype=np.float32)
#         self.numpy_B[index] = np.asarray( [ item for item in jsProgram['b2'] ], dtype=np.float32)
#         self.numpy_A[index] = np.asarray(jsProgram['a2'], dtype=np.float32)
#
#         self.programs[index] = decProg
#
#
#     def topKProgs(self, k=10):
#         topKids = (-self.distance).argsort()[:k] # slow to get topK
#         topKProgs = [self.programs[_id] for _id in topKids]
#         return topKProgs
#
#
#     def measureDistance(self, embedding):
#
#
#         # a1 is np.scalar or (1)
#         # b1 is shaped (latent_size)
#         a1 = embedding.A
#         b1 = embedding.B
#
#         a2 = self.numpy_A
#         b2 = self.numpy_B
#         probY = self.numpy_ProbY
#
#
#         latent_size = len(b1)
#
#         a_star = a1 + a2 + 0.5 # shape is [numItemsInDB]
#         b_star = np.expand_dims(b1,axis=0) + b2 # shape is [numItemsInDB , latent_size]
#
#         ab1 = np.sum(np.square(b1)/(4*a1), axis=0) + 0.5 * latent_size * np.log(-a1/np.pi) # shape is ()
#         ab2 = np.sum(np.square(b2)/(4*np.tile(np.expand_dims(a2,1), [1,latent_size])), axis=1) \
#                                 + 0.5 *  latent_size * np.log(-a2/np.pi) # shape is [numItemsInDB]
#         ab_star = np.sum(np.square(b_star)/(4* np.tile(np.expand_dims(a_star,1), [1,latent_size])), axis=1) \
#                                 + 0.5 *  latent_size * np.log(-a_star/np.pi) # shape is [numItemsInDB]
#         cons = 0.5 * latent_size * np.log( 2*np.pi )
#
#         prob = ab1 + ab2 - ab_star - cons + probY
#
#         self.distance = prob
#         return
#
#






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


    def topKProgs(self, k=10):

        topKforBatch = []
        for item in range(self.batch_size):
            topKids = (-self.distance[item]).argsort()[:k] # slow to get topK
            topKProgs = [self.programs[_id] for _id in topKids]
            topKforBatch.append(topKProgs)

        return topKforBatch


    def measureDistance(self, embedding):

        #a1 = embedding.A  # a1 is (batch_size)
        #b1 = embedding.B  # b1 is shaped (batch_size, latent_size)

        #a2 = self.numpy_A # [num_items]
        #b2 = self.numpy_B # [num_items , latent_size]
        #probY = self.numpy_ProbY # [num_items]

        latent_size = np.shape(embedding.B)[1]

        #a_star = np.tile(np.expand_dims(a1, axis=1),  [1, self.numItems])      +  np.tile( np.expand_dims(a2 , axis=0),  [self.batch_size, 1]) + 0.5 # shape is [batch_size , numItems]
        a_star =  embedding.A[:,None] + self.numpy_A[None, :]  + 0.5 # shape is [batch_size , numItems]
        #b_star = np.tile(np.expand_dims(b1, axis=1) , [1, self.numItems , 1] ) +  np.tile( np.expand_dims(b2 , axis=0) , [self.batch_size, 1, 1] )   # shape is [batch_size, numItems, latent_size]
        b_star = embedding.B[:,None,:] + self.numpy_B[None, :, :]   # shape is [batch_size, numItems, latent_size]

        #ab1 = np.sum(np.square(b1)/(4*np.tile(np.expand_dims(a1,axis=1) , [1, latent_size])), axis=1) + 0.5 * latent_size * np.log(-a1/np.pi) # shape is (batch_size)
        ab1 = np.sum( (np.square(embedding.B) / (4*embedding.A[:,None]) ), axis=1) + 0.5 * latent_size * np.log(-embedding.A / np.pi) # shape is (batch_size)
        #ab1 = np.tile(np.expand_dims(ab1, axis=1),[1, self.numItems]) # shape is (batch_size , numItems)

        ab2 = np.sum( (np.square(self.numpy_B) / (4*self.numpy_A[:, None]) ), axis=1) + 0.5 * latent_size * np.log(-self.numpy_A/np.pi) # shape is [numItems]
        #ab2 = np.tile(np.expand_dims(ab2, axis=0),[self.batch_size, 1]) # shape is (batch_size , numItems)

        #ab_star = np.sum(np.square(b_star)/(4* np.tile(np.expand_dims(a_star,2), [1,1,latent_size])), axis=2) + 0.5 *  latent_size * np.log(-a_star/np.pi) # shape is [batch_size , numItems]
        ab_star = np.sum((np.square(b_star) /  (4 * a_star[:,:,None])), axis=2) + 0.5 *  latent_size * np.log(-a_star/np.pi) # shape is [batch_size , numItems]

        cons = 0.5 * latent_size * np.log( 2*np.pi )
        #probY = np.tile(np.expand_dims(probY, axis=0),[self.batch_size, 1])

        self.distance = ab1[:,None] + ab2[None,:] - ab_star - cons + self.numpy_ProbY[None,:]

        return


    # def measureDistance2(self, embedding):
    #
    #     a1 = embedding.A  # a1 is (batch_size)
    #     b1 = embedding.B  # b1 is shaped (batch_size, latent_size)
    #
    #     a2 = self.numpy_A # [num_items]
    #     b2 = self.numpy_B # [num_items , latent_size]
    #     probY = self.numpy_ProbY # [num_items]
    #
    #     latent_size = np.shape(embedding.B)[1]
    #
    #     a_star = np.tile(np.expand_dims(a1, axis=1),  [1, self.numItems])      +  np.tile( np.expand_dims(a2 , axis=0),  [self.batch_size, 1]) + 0.5 # shape is [batch_size , numItems]
    #     # a_star =  embedding.A[:,None] + self.numpy_A[None, :]   # shape is [batch_size , numItems]
    #     b_star = np.tile(np.expand_dims(b1, axis=1) , [1, self.numItems , 1] ) +  np.tile( np.expand_dims(b2 , axis=0) , [self.batch_size, 1, 1] )   # shape is [batch_size, numItems, latent_size]
    #     # b_star = embedding.B[:,None,:] + self.numpy_B[None, :, :]   # shape is [batch_size, numItems, latent_size]
    #
    #     ab1 = np.sum(np.square(b1)/(4*np.tile(np.expand_dims(a1,axis=1) , [1, latent_size])), axis=1) + 0.5 * latent_size * np.log(-a1/np.pi) # shape is (batch_size)
    #     # ab1 = np.sum( (np.square(embedding.B).T / (4*embedding.A) ).T, axis=1) + 0.5 * latent_size * np.log(-embedding.A / np.pi) # shape is (batch_size)
    #     ab1 = np.tile(np.expand_dims(ab1, axis=1),[1, self.numItems]) # shape is (batch_size , numItems)
    #
    #     # ab2 = np.sum( (np.square(self.numpy_B).T / (4*self.numpy_A) ).T, axis=1) + 0.5 * latent_size * np.log(-self.numpy_A/np.pi) # shape is [numItems]
    #
    #     ab2 = np.sum(np.square(b2)/(4*np.tile(np.expand_dims(a2,axis=1), [1,latent_size])), axis=1)  + 0.5 *  latent_size * np.log(-a2/np.pi) # shape is [num_items]
    #     ab2 = np.tile(np.expand_dims(ab2, axis=0),[self.batch_size, 1]) # shape is (batch_size , numItems)
    #
    #     ab_star = np.sum(np.square(b_star)/(4* np.tile(np.expand_dims(a_star,2), [1,1,latent_size])), axis=2) + 0.5 *  latent_size * np.log(-a_star/np.pi) # shape is [batch_size , numItems]
    #     # ab_star = np.sum(b_star/a_star[:,:,None], axis=2) + 0.5 *  latent_size * np.log(-a_star/np.pi) # shape is [batch_size , numItems]
    #
    #     cons = 0.5 * latent_size * np.log( 2*np.pi )
    #     #probY = np.tile(np.expand_dims(probY, axis=0),[self.batch_size, 1])
    #
    #     self.distance = ab1 + ab2 - ab_star - cons + probY
    #     # self.distance = ab1[:,None] + ab2[None,:] - ab_star - cons + self.numpy_ProbY[None,:]
    #
    #     return


    # def measureDistance(self, embedding):
    #
    #     B_a1 = embedding.A  # a1 is (batch_size)
    #     B_b1 = embedding.B  # b1 is shaped (batch_size, latent_size)
    #
    #     B_a2 = self.numpy_A # [num_items]
    #     B_b2 = self.numpy_B # [num_items , latent_size]
    #     B_probY = self.numpy_ProbY # [num_items]
    #     # all inputs are np.arrays
    #     batch_size = np.shape(embedding.B)[0]
    #     latent_size = np.shape(embedding.B)[1]
    #
    #     probssss= []
    #     for i in range(batch_size):
    #         a1 = B_a1[i]
    #         b1 = B_b1[i]
    #         a2 = B_a2
    #         b2 = B_b2
    #         prob_Y = B_probY
    #
    #         a_star = a1 + a2 + 0.5  # shape is [num_items]
    #         b_star = np.expand_dims(b1,axis=0) + b2  # shape is [num_items, latent_size]
    #
    #         ab1 = np.sum(np.square(b1)/(4*a1), axis=0) + 0.5 * latent_size * np.log(-a1/np.pi) # shape is ()
    #         ab2 = np.sum(np.square(b2)/(4*np.tile(np.expand_dims(a2,1), [1,latent_size])), axis=1) \
    #                             + 0.5 *  latent_size * np.log(-a2/np.pi) # shape is [num_items]
    #         ab_star = np.sum(np.square(b_star)/(4* np.tile(np.expand_dims(a_star,1), [1,latent_size])), axis=1) \
    #                             + 0.5 *  latent_size * np.log(-a_star/np.pi) # shape is [num_items]
    #         cons = 0.5 * latent_size * np.log( 2*np.pi )
    #
    #         prob = ab1 + ab2 - ab_star - cons + prob_Y
    #
    #         probssss.append(prob)
    #
    #     self.distance = np.asarray(probssss)
    #     return
