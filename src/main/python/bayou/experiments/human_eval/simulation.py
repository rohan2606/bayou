import numpy as np
import pandas as pd

from copy import deepcopy
from data import arka, akash, sushovan, swarna, binhang, sourav, meghana, dimitri, raj, letao, yuxin

NUM_USERS = 11
NUM_USERS_EXP = 1

NUM_EXAMPLES = 15
NUM_METHODS = 4
NUM_TRIALS = 100000


#H_0^{A,B} = not (An arbitrary user is more likely to prefer method A 
#to method B)
#H_0^{A,B} = An arbitrary user is not more likely to prefer method A to method B
#H_0^{A,B} = For an arbitrary user method B is at least as good method A
#H_0^{Rohan's,CodeHow} = For an arbitrary user CodeHow is at least as good Rohan's -> 0.0 (can be rejected) 
#H_0^{Rohan's,Rohan's} = For an arbitrary user Rohan's is at least as good Rohan's = 1.0 (cannot be rejected)
#H_0^{CodeHow,Rohan's} = For an arbitrary user Rohan's is at least as good CodeHow's -> 1 (cannot be rejected) 
#
#So:
#
#H_0^{Rohan's, CodeHow} = An arbitrary user is not more likely to 
#prefer Rohan's method to CodeHow


class User:

    def __init__(self, name, expertise, values):
        self.name = name
        self.expertise = expertise
        self.values = values

    def get_problem_ratings(self, method_id, problem_id):
        return self.values[problem_id][method_id]



class Problem:
    def __init__(self, name, id):
        self.name = name
        self.id = id

class Method:
    def __init__(self, name, id):
        self.name = name
        self.id = id



class Hypothesis_Testing:

    def __init__(self):
        self.problem_names   =   ['Client', 'Crypto', 'FileUtils', 'gen_list' ,'GuiAppl',
                              'hashToArray', 'IO', 'Palindrome', 'Parser', 'PeekingIt',
                              'SQL_select', 'Stopwatch', 'ThreadQ', 'WordCount', 'XMLUtils'
                            ]

        self.method_names  =    ['Codex', 'Non_Prob', 'DeepCodeSearch', 'CodeHow' ]
        self.user_names =['arka', 'akash', 'sushovan', 'swarna', 'binhang', 'sourav', 'meghana', 'dimitri', 'raj', 'letao', 'yuxin']
        self.user_data = [arka, akash, sushovan, swarna, binhang, sourav, meghana, dimitri, raj, letao, yuxin]


        self.list_of_problems = self.define_problems()
        self.list_of_users = self.define_users()
        self.list_of_methods = self.define_methods()

        self.get_vital_stats()


    def get_vital_stats(self):
        all_data = np.array(self.user_data)

        mean_score = np.mean(all_data, axis=(0,1))
        print(mean_score)


    def define_problems(self):


        list_of_problems = list()
        for id in range(NUM_EXAMPLES):
            problem = Problem( self.problem_names[id] , id)
            list_of_problems.append(problem)

        return list_of_problems

    def define_users(self):
        list_of_users = list()
        for user_num in range(NUM_USERS):
            user = User(self.user_names[user_num], 1. , self.user_data[user_num])
            list_of_users.append(user)
        return list_of_users

    def define_methods(self):
        list_of_methods = list()
        for method_num in range(NUM_METHODS):
            method = Method(self.method_names[method_num] ,method_num)
            list_of_methods.append(method)
        return list_of_methods

    def get_random_problem(self):
        idx = np.random.randint(low=0, high=len(self.list_of_problems))
        return idx, self.list_of_problems[idx]

    def get_random_user(self):
        idx = np.random.randint(low=0, high=len(self.list_of_users))
        return idx, self.list_of_users[idx]

    def methodA_rated_better_than_methodB(self, user, methodA_id, methodB_id, problem_id):
        score1 = user.get_problem_ratings(methodA_id, problem_id)
        score2 = user.get_problem_ratings(methodB_id, problem_id)
        if score2 > score1:
            return -1
        elif score2 < score1:
            return 1
        else:
            return 0


    def get_A_wins_this_person(self, userP, methodA, methodB):
        AWinsThisPerson = 0;
        for j in range(NUM_EXAMPLES): #1 to 15:
          id, userP = self.get_random_user()
          problem_id, problem = self.get_random_problem()
          AIsBetter = self.methodA_rated_better_than_methodB(userP, methodA.id, methodB.id, problem_id)
          AWinsThisPerson = AWinsThisPerson + AIsBetter

        if AWinsThisPerson > 0:
            return True #BWinsThisTime++

    def get_A_wins_this_time(self, methodA, methodB):
        AWinsThisTime = 0
        for i in range(NUM_USERS_EXP): #1 to 10:
            id, userP = self.get_random_user()
            didAwin = self.get_A_wins_this_person(userP, methodA, methodB)
            if didAwin:
                AWinsThisTime = AWinsThisTime + 1

        if AWinsThisTime > NUM_USERS_EXP/2:
            return True #BWinsThisTime #MethodAAsGood++



    def get_p_value_AtoB(self, methodA, methodB):
        MethodBAsGood = 0
        for attempt in range(NUM_TRIALS):
            AIsBetter = self.get_A_wins_this_time(methodA, methodB)
            if not AIsBetter:
                 MethodBAsGood = MethodBAsGood+1

        p_value = MethodBAsGood/NUM_TRIALS
        return p_value


    def get_p_value_matrix(self):
        p_value_matrix = [[None for _ in range(NUM_METHODS)] for _ in range(NUM_METHODS)]

        for i, a in enumerate(self.list_of_methods):
            for j, b in enumerate(self.list_of_methods):
                print(i,j)
                p_value_matrix[i][j] = self.get_p_value_AtoB(a, b)

        return p_value_matrix


def transpose_correlation_matrix(matrixA, id1, id2, col_names):
    temp = deepcopy(matrixA[id2])
    matrixA[id2] = matrixA[id1]
    matrixA[id1] = temp

    temp = deepcopy(matrixA[:,id2])
    matrixA[:,id2] = matrixA[:,id1]
    matrixA[:,id1] = temp

    temp = col_names[id2]
    col_names[id2] = col_names[id1]
    col_names[id1] = temp
    return matrixA, col_names


mat  = Hypothesis_Testing().get_p_value_matrix()
np_mat = np.matrix(mat)
cols = ["CODEC", "Non-Prob Codec", "Deep Code Search", "CodeHow"]
#print(pd.DataFrame(np_mat, index=cols,columns=cols))

np_mat, cols = transpose_correlation_matrix(np_mat, 1, 2, cols)
np_mat, cols = transpose_correlation_matrix(np_mat, 2, 3, cols)
pd.set_option("display.precision", 8)
print(pd.DataFrame(np_mat, index=cols,columns=cols))
