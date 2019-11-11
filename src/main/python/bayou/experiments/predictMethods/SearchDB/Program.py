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

        self.ret_type = programJson['rt'] if 'rt' in programJson else None
        self.form_param = programJson['fp'] if 'fp' in programJson else None
        # self.distance = [np.inf for i in range(batch_size)]


    def stripJavaDoc(self, stringBody):
        return re.sub(r'/\*\*(.*?)\*\/', '', stringBody.replace('\n',''))
