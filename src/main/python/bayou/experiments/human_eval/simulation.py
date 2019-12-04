import numpy as np
np.set_printoptions(precision=10)

from data import arka, akash, sushovan, swarna, binhang, sourav, meghana, dimitri, raj, letao, yuxin

NUM_USERS = 10
NUM_USERS_EXP = 1

NUM_EXAMPLES = 15
NUM_METHODS = 4
NUM_TRIALS = 100000


# For an arbitrarily selected programmer, method A is not better than method B
# In other words, For an arbitrarily selected programmer, method A is as good as method B
# Alternate hypothesis, For an arbitrarily selected programmer, method B is better


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
        self.user_names =['arka', 'akash', 'sushovan', 'swarna', 'binhang', 'sourav', 'meghana', 'dimitri', 'raj', 'letao']
        self.user_data = [arka, akash, sushovan, swarna, binhang, sourav, meghana, dimitri, raj, letao]


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

    def methodB_rated_better_than_methodA(self, user, methodA_id, methodB_id, problem_id):
        score1 = user.get_problem_ratings(methodA_id, problem_id)
        score2 = user.get_problem_ratings(methodB_id, problem_id)
        if score2 > score1:
            return 1
        elif score2 < score1:
            return -1
        else:
            return 0


    def get_B_wins_this_person(self, userP, methodA, methodB):
        BWinsThisPerson = 0;
        for j in range(NUM_EXAMPLES): #1 to 15:
          id, userP = self.get_random_user()
          problem_id, problem = self.get_random_problem()
          BIsBetter = self.methodB_rated_better_than_methodA(userP, methodA.id, methodB.id, problem_id)
          BWinsThisPerson = BWinsThisPerson + BIsBetter

        if BWinsThisPerson > 0:
            return True #BWinsThisTime++

    def get_B_wins_this_time(self, methodA, methodB):
        BWinsThisTime = 0
        for i in range(NUM_USERS_EXP): #1 to 10:
            id, userP = self.get_random_user()
            didBwin = self.get_B_wins_this_person(userP, methodA, methodB)
            if didBwin:
                BWinsThisTime = BWinsThisTime + 1

        if BWinsThisTime > NUM_USERS_EXP/2:
            return True #BWinsThisTime #MethodAAsGood++



    def get_p_value_AtoB(self, methodA, methodB):
        MethodAAsGood = 0
        for attempt in range(NUM_TRIALS):
            BIsBetter = self.get_B_wins_this_time(methodA, methodB)
            if not BIsBetter:
                 MethodAAsGood = MethodAAsGood+1

        p_value = MethodAAsGood/NUM_TRIALS
        return p_value


    def get_p_value_matrix(self):
        p_value_matrix = [[None for _ in range(NUM_METHODS)] for _ in range(NUM_METHODS)]

        for i, a in enumerate(self.list_of_methods):
            for j, b in enumerate(self.list_of_methods):
                print(i,j)
                p_value_matrix[i][j] = self.get_p_value_AtoB(a, b)

        return p_value_matrix



ps  = Hypothesis_Testing().get_p_value_matrix()
print(np.matrix(ps))
