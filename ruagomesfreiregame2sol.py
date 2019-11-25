import random
import numpy as np
import math

# LearningAgent to implement
# no knowledge about the environment can be used
# the code should work even with another environment
class LearningAgent:

        # init
        # nS maximum number of states
        # nA maximum number of action per state
        def __init__(self,nS,nA):

                self.nS = nS
                self.nA = nA

                self.alpha = 0.5 #Learning rate
                self.gamma = 0.3 #Discount rate
                self.epsilon = 0.3 #Exploration/Exploitation balance

                self.possibleActions = 0

                #initialize matrix with number of states per number of actions per state 
                #(all with value zero to identify if it has been already tested or not)
                #we will update the value to another if it was already tested
                # self.Q = [nS] * [nA] -> initialize with zero
                
                self.Q = np.array([[ 0.0 for i in range(nA)] for j in range(nS)])
                
              
        
        # Select one action, used when learning  
        # st - is the current state        
        # aa - is the set of possible actions
        # for a given state they are always given in the same order
        # returns
        # a - the index to the action in aa
        def selectactiontolearn(self,st,aa):
                # define this function
                # print("select one action to learn better")

                #change epsilon with time
                
                a = 0
                value = self.Q[st][a]
                self.possibleActions = len(aa)
                
                #random.uniform(0, 1) = current epsilon
                #if current epsilon < self.epsilon then selected a random action from aa or random undiscovered from aa(if there is one)
                #else selected action with best reward 

                #NOTA
                #Matrix that saves information about the frequency of each action
                #No need to use self.epsilon

                if random.uniform(0, 1) < self.epsilon:
                        randomAction = random.choice(aa) 
                        for i in aa:
                                if randomAction == i:
                                        break
                                a+=1
                else:
                        for i in range(len(aa)):
                                if value < self.Q[st][i]:
                                        value = self.Q[st][i]  
                                        a = i 
                # for i in range(len(aa)):
                #         if value < self.Q[st][i]:
                #                 value = self.Q[st][i]  
                #                 a = i 
                                                         
                return a

        # Select one action, used when evaluating
        # st - is the current state        
        # aa - is the set of possible actions
        # for a given state they are always given in the same order
        # returns
        # a - the index to the action in aa
        def selectactiontoexecute(self,st,aa):
                # define this function
                # print("select one action to see if I learned")

                #Go to matrix and find the action with the best reward to execute 
                #max = np.max(Q[st, :])
                #max in possible actions (aa)
                a = 0
                value = self.Q[st][a]
                for i in range(len(aa)):
                        if value < self.Q[st][i]:
                                value = self.Q[st][i]  
                                a = i  
                return a


        # this function is called after every action
        # ost - original state
        # nst - next state
        # a - the index to the action taken
        # r - reward obtained
        def learn(self,ost,nst,a,r):
                #print("learn something from this data")

                #finds max of Q[y][b] (row of ost)
                max = -math.inf
                for i in range(self.possibleActions):
                        if self.Q[nst][i] > max:
                                max = self.Q[nst][i]

                #max = np.max(self.Q[nst, :])
                #updates quality in Q 
                self.Q[ost,a] = self.Q[ost,a] + self.alpha*(r + self.gamma*max - self.Q[ost,a])

