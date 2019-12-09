import math
import numpy as np
import re

class skinnyProgramInBatch():

    def __init__(self, programJson, index2ColDB, batch_size):
        self.fileName = programJson['file']
        self.methodName = programJson['method']
        self.body = self.stripJavaDoc(programJson['body'])

        self.batch_size = batch_size
        self.index = index2ColDB

        # self.distance = [np.inf for i in range(batch_size)]

    def stripJavaDoc(self, stringBody):
        temp = re.sub(r'/\*\*(.*?)\*\/', '', stringBody.replace('\n','') )
        temp = ' '.join([ word for word in temp.split() if not word.startswith('@') ])
        temp = temp.replace('private', 'public')
        return temp

