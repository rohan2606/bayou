from multiprocessing import Pool
import ijson.backends.yajl2_cffi as ijson
import simplejson as json
from MyDataBase import MyColumnDatabaseWBatch
from Program import skinnyProgramInBatch

class parallelReadJSON():


    def __init__(self, folder, numThreads, dimension=256, batch_size=50, minJSONs = 1, maxJSONs = 70):
        self.folder = folder
        self.minJSONs = minJSONs
        self.maxJSONs = maxJSONs
        self.dimension = dimension
        self.batch_size = batch_size
        self.numThreads = numThreads


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
                decodedProgram = skinnyProgramInBatch(jsProgram, k, Program_DB_j, self.batch_size)
                Program_DB_j.setValues(jsProgram, decodedProgram, k)

        # print("Read " + fileName)
        return Program_DB_j
