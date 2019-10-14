import json
import numpy as np


class EmbeddingBatch():

    def __init__(self, batchProgram, batch_size, dimension):

        self.psi = np.zeros([batch_size, dimension] , dtype=np.float32 )
        self.js = [None for i in range(batch_size)]

        for j, program in enumerate(batchProgram):
            self.psi[j] = np.asarray(program['ev_psi'])
            del program['ev_psi']
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
                    batchProgram = []
        return
