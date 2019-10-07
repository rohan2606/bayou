from multiprocessing import Pool

from bayou.experiments.predictMethods.SearchDB.MyDataBase import MyColumnDatabaseWBatch
from bayou.experiments.predictMethods.SearchDB.Program import skinnyProgramInBatch

import ijson.backends.yajl2_cffi as ijson
import simplejson as json
import pickle
import os

backupDB = '../log/bayouSearchColDb_backup.pkl'

class parallelReadJSON():


    def __init__(self, folder, numThreads, dimension=256, batch_size=50, minJSONs = 0, maxJSONs = 70):
        self.folder = folder
        self.minJSONs = minJSONs
        self.maxJSONs = maxJSONs
        self.dimension = dimension
        self.batch_size = batch_size
        self.numThreads = numThreads


    def getSearchDatabase(self):

        if not os.path.exists(backupDB):
            FinalProgram_DB = self.readAllJSONs()
            #with open(backupDB , 'wb') as output:
            #    pickle.dump(FinalProgram_DB, output)
        else:
            with open(backupDB, 'rb') as input:
                FinalProgram_DB = pickle.load(input)

        return FinalProgram_DB



    def readAllJSONs(self):

        prefix = self.folder + 'Program_output_'
        files = [ prefix + str(j) + '.json' for j in range(self.minJSONs, self.maxJSONs)]


        numThreads = self.numThreads
        print("Start parallel multiprocessing read JSONs")
        fileChunks = [files[i::numThreads] for i in range(numThreads)]
        pool = Pool(processes=numThreads)
        result = pool.map(self.readMultipleJSONs, fileChunks)

        # pool.close()
        # pool.join()
        print("Done with multi-multiprocessing read JSONs")
        #Aggregate the result
        FinalProgram_DB = []
        for item in result:
            FinalProgram_DB.extend(item)

        return FinalProgram_DB



    def readMultipleJSONs(self, files):

        Program_DB_all=[]
        #print ("Starting to read " + str(len(files)) + " JSON files")
        for file in files:
            Program_DB_j = self.readEachJSON(file)
            Program_DB_all.append(Program_DB_j)


        #print ("Completed reading " + str(len(files)) + " JSON files")
        return Program_DB_all


    def readEachJSON(self, fileName):
        # print("Starting to read " + fileName)

        with open( fileName , 'r') as f:
            js = json.load(f)
            numItems = len(js['programs'])

            Program_DB_j = MyColumnDatabaseWBatch( numItems , self.dimension, self.batch_size)

            for k, jsProgram in enumerate(js['programs']):
                decodedProgram = skinnyProgramInBatch(jsProgram, k, self.batch_size)
                Program_DB_j.setValues(jsProgram, decodedProgram, k)

        # print("Read " + fileName)
        return Program_DB_j
