import numpy as np 

class Perceptron:
    def __init__(self , N , alpha=0.1):#N is number of columns in our input feature vector
        # initialize the weight matrix and store the learning rate
        self.W = np.random.randn(N+1) / np.sqrt(N)
        self.alpha = alpha

    def step(self , x):
        #apply the step function
        return 1 if x > 0 else 0