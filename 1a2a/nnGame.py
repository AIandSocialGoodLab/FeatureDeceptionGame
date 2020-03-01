from nn import NN
import random
import numpy as np
import torch
class Game:
    def __init__(self, seed, n, K, K2, K3):
        self.n = n
        self.K = K
        self.K2 = K2
        self.K3 = K3
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.xhat = np.random.rand(n, K)
        self.us = np.random.rand(n)
        self.costs = np.random.rand(n, K)
        self.budget = np.random.rand() * n * K * 0.2
        self.f = NN(K, K2, K3, False)

    def printGame(self):
        for i in range(self.n):
            print("Node", i )
            self.nodes[i].printFeatures()

    def getInitialXhats(self):
        xhat = -1 * np.ones((self.n, self.K))
        for i in range(self.n):
            for k in range(self.K):
                xhat[i,k] = self.nodes[i].features[k].xhat
        return xhat

    def getFeatureRanges(self):
        ranges = -1 * np.ones((self.n, self.K))
        for i in range(self.n):
            for k in range(self.K):
                ranges[i,k] = self.nodes[i].features[k].range
        return ranges
