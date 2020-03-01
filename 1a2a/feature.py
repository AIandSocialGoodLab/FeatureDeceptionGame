import random
import numpy as np

class Feature:
    def __init__(self, xhat, discrete, cost, range, weight):
        self.xhat = xhat
        self.discrete = discrete
        self.cost = cost
        self.range = range
        self.weight = weight

    def __init__(self, xhat, discrete, weight):
        self.xhat = xhat
        self.discrete = discrete
        self.cost = None
        self.range = None
        self.weight = weight

    def __init__(self, discrete, seed):
        self.discrete = discrete
        self.weight = np.random.rand()*2 - 1
        if self.discrete:
            self.xhat = np.random.randint(0, 1)
            self.cost = np.random.rand() * 6 - 3
            self.range = None
        else:
            self.xhat = np.random.rand()
            self.cost = np.random.rand() * 3
            self.range = np.random.rand() / 4
