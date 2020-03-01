from feature import Feature
from itertools import product
import numpy as np
import random
class Node:
    def __init__(self, K, Cweights, Dweights, seed):
        self.K = K
        self.seed = seed
        self.Kd = int(K*2/3)
        self.Kc = int(K*1/3)
        self.Cfeatures = [Feature(False, seed) for k in range(self.Kc)]
        self.Dfeatures = [Feature(True, seed) for k in range(self.Kd)]
        self.CfeatureWeights = Cweights
        self.DfeatureWeights = Dweights
        self.group = np.arange(self.Kd).reshape((2,-1))
        for i in range(self.Kc):
            self.Cfeatures[i].weight = self.CfeatureWeights[i]
        for i in range(self.Kd):
            self.Dfeatures[i].weight = self.DfeatureWeights[i]
        for i in range(self.group.shape[0]):
            one = np.random.randint(self.group[i,0], self.group[i,0]+self.group.shape[1])
            for k in self.group[i,:]:
                self.Dfeatures[k].xhat = 0
            self.Dfeatures[one].xhat = 1
        self.u = np.random.rand()

    def getAlpha(self):
        alpha = 0
        for f in self.features[self.Kd:]:
            alpha += min(f.weight * max(f.xhat - f.range, 0), f.weight * min(f.xhat + f.range, 1))
        alpha = np.exp(alpha)
        return alpha

    def getBeta(self):
        beta = 0
        for f in self.features[self.Kd:]:
            beta += max(f.weight * max(f.xhat - f.range, 0), f.weight * min(f.xhat + f.range, 1))
        beta = np.exp(beta)
        return beta


    def getMaxCost(self):
        maxCost = 0
        for f in self.Cfeatures:
            maxCost += f.cost * min(f.xhat, f.range, 1-f.xhat)
        for f in self.Dfeatures:
            maxCost += f.cost
        return maxCost

    def printFeatures(self):
        for f in self.features:
            feat = {"xhat": f.xhat, "cost": f.cost, "range": f.range, "weight": f.weight}
            print(feat)

    def generateCorrelation(self):
        K = 10
        corr = [3, 2, 2]
        l = sum(corr)
        m1 = [(1,0,0), (0,1,0), (0,0,1)]
        m2 = [(1,0), (0,1)]
        m3 = m2
        m4 = list(product(range(2), repeat=K-l))
        m = list(product(m1, m2, m3, m4))
        self.discreteSpace = [i[0] + i[1] + i[2] + i[3] for i in m]
        self.discreteCost = [random.random()*10 for i in self.discreteSpace]
        ws = np.array([f.weight for f in self.features[:self.Kd]])
        self.discreteScore = [np.exp(np.dot(ws, np.array(i))) for i in self.discreteSpace]
