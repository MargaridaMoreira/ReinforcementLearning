import random

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

                self.alpha = 0.9 #Learning rate
                self.gamma = 0.75 #Discount rate
                self.epsilon = 0.2 #Exploration/Exploitation balance

                #initialize matrix with number of states per number of actions per state 
                #(all with value "none" to identify if it has been already tested or not)
                #we will update the value to a number if it was already tested
                
              
        
        # Select one action, used when learning  
        # st - is the current state        
        # aa - is the set of possible actions
        # for a given state they are always given in the same order
        # returns
        # a - the index to the action in aa
        def selectactiontolearn(self,st,aa):
                # define this function
                # print("select one action to learn better")

                #random.uniform(0, 1) = current epsilon
                #if current epsilon < self.epsilon then selected a random undiscovered action (none)
                #else selected action with best reward 

                a = 0
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

                a = 0
                return a


        # this function is called after every action
        # ost - original state
        # nst - next state
        # a - the index to the action taken
        # r - reward obtained
        def learn(self,ost,nst,a,r):
                # define this function
                #print("learn something from this data")

                #finds max of Q[y][b] (row of ost)
                #

                #updates reward of matrix 
                #Q[ost][a] = Q[ost][a] + self.alpha*(r + self.gamma*max - Q[ost][a])
                
                return
