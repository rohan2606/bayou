import math
import numpy as np


# class skinnyProgram():
#
#     def __init__(self, programJson, index2ColDB, colDB):
#         self.fileName = programJson['file']
#         self.methodName = programJson['method']
#         self.body = programJson['body']
#
#         self.index = index2ColDB
#
#         self.distance = np.inf
#         self.colDB = colDB
#
#
#     def print_self(self, rank):
#         print("Rank :: " , rank)
#         print("Prob :: " , self.distance)
#         print("Body :: " , self.body)
#
#
#     def getDistance(self):
#         self.distance = self.colDB.distance[self.index]
#         return


class skinnyProgramInBatch():

    def __init__(self, programJson, index2ColDB, colDB, batch_size):
        self.fileName = programJson['file']
        self.methodName = programJson['method']
        self.body = programJson['body']

        self.batch_size = batch_size
        self.index = index2ColDB

        self.distance = [np.inf for i in range(batch_size)]
        self.colDB = colDB


    def print_self(self, rank):
        print("Rank :: " , rank)
        print("Prob :: " , self.distance)
        print("Body :: " , self.body)


    def getDistance(self, batch_id):
        self.distance[batch_id] = self.colDB.distance[batch_id, self.index]
        return

# class Program():
#
#     self.distance = -math.inf
#     #
#     # def Program(self, fileName, methodName, body, ProbY, B2, A2):
#     #     self.fileName = fileName
#     #     self.methodName = methodName
#     #     self.body = body
#     #
#     #     self.ProbY = ProbY
#     #     self.B = B
#     #     self.A = A
#
#     def Program(self, programJson):
#         self.fileName = program['file']
#         self.methodName = program['method']
#         self.body = program['body']
#
#         self.ProbY = np.asarray(program['ProbY'] , dtype=np.float64)
#         self.B = [ np.asarray(item, dtype=np.float64) for item in program['b2'] ]
#         self.A = np.asarray(program['a2'], dtype=np.float64)
#
#
#     def set_distance(self, distance):
#         self.distance = distance
#
#     def get_distance(self):
#         return self.distance
#
#     def reset_distance(self):
#         self.distance = -math.inf
#
#
#     def measureDistance(self, embedding):
#
#         a1 = embedding.A
#         b1 = embedding.B
#
#         a2 = self.A
#         b2 = self.B
#         probY = self.ProbY
#
#
#         latent_size = np.shape(a1)[0]
#
#         a_star = a1 + a2 + 0.5 # shape is [batch_size]\
#         b_star = b1 + b2
#
#         ab1 = np.sum(np.square(b1)/(4*a1)) + 0.5 * latent_size * np.log(-a1/np.pi)
#         ab2 = np.sum(np.square(b2)/(4*a2)) + 0.5 *  latent_size * np.log(-a2/np.pi)
#         ab_star = np.square(b_star)/(4*a_star) + 0.5 *  latent_size * np.log(-a_star/np.pi)
#         cons = 0.5 * latent_size * np.log( 2*np.pi )
#
#         prob = ab1 + ab2 - ab_star - cons + probY
#
#         self.distance = prob
#         return
#
