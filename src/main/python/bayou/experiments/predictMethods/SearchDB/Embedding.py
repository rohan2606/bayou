import json
import numpy as np


class EmbeddingBatch():

    def __init__(self, batchProgram, batch_size, dimension):

        self.A = np.zeros([batch_size] , dtype=np.float32)
        self.B = np.zeros([batch_size, dimension] , dtype=np.float32 )
        self.js = [None for i in range(batch_size)]

        for j, program in enumerate(batchProgram):
            self.A[j] = np.asarray(program['a1'])
            self.B[j] = np.asarray(program['b1'])
            del program['a1']
            del program['b1']
            self.js[j] = program


class Embedding_iterator_WBatch():


    def __init__(self, file, batch_size, dimension):
        self.embList = []
        self.maxCount = 0
        self.batch_size = batch_size
        self.dimension = dimension
        with open(file, 'r') as f:
            js = json.load(f)
            batchProgram = []
            for program in js['embeddings']:
                batchProgram.append(program)
                self.maxCount += 1
                if self.maxCount % batch_size == 0:
                    emb = EmbeddingBatch(batchProgram, self.batch_size, self.dimension)
                    self.embList.append(emb)
                    #print("Length of batch prog is " + str(len(batchProgram)))
                    batchProgram = []
        # self.iterator = 0 # Iter has to start from 1 since maxCount does not count from 0
        return
