from itertools import product
import numpy as np
import random
class nnNode:
    def __init__(self, K, Cweights, Dweights, seed):
        self.K = K
        self.seed = seed
        self.u = np.random.rand()


    def printFeatures(self):
        for f in self.features:
            feat = {"xhat": f.xhat, "cost": f.cost, "range": f.range, "weight": f.weight}
            print(feat)
