import math
import numpy as np
import re

class skinnyProgramInBatch():

    def __init__(self, programJson, index2ColDB, colDB, batch_size):
        #self.fileName = programJson['file']
        #self.methodName = programJson['method']
        self.body = self.stripJavaDoc(programJson['body'])

        self.batch_size = batch_size
        self.index = index2ColDB

        self.distance = [np.inf for i in range(batch_size)]
        self.colDB = colDB


    def stripJavaDoc(self, stringBody):
        return re.sub(r'/\*\*(.*?)\*\/', '', stringBody.replace('\n',''))

    def print_self(self, rank):
        print("Rank :: " , rank)
        print("Prob :: " , self.distance)
        print("Body :: " , self.body)


    def setDistance(self, batch_id):
        self.distance[batch_id] = self.colDB.distance[batch_id, self.index]
        return

    def getDistance(self, batch_id):
        return self.distance[batch_id]
