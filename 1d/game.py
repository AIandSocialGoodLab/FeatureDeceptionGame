from node import Node
import random
import numpy as np
class Game:
    def __init__(self, seed, n, K):
        self.n = n
        self.K = K
        self.Kd = int(K*2/3)
        self.Kc = int(K*1/3)
        self.seed = seed
        np.random.seed(seed)
        self.CfeatureWeights = np.random.rand(self.Kc)-0.5
        self.DfeatureWeights = np.random.rand(self.Kd)-0.5
        self.nodes = [Node(K, self.CfeatureWeights, self.DfeatureWeights, seed) for i in range(n)]
        self.us = np.array([node.u for node in self.nodes])
        maxCost = 0
        for i in range(n):
            maxCost += self.nodes[i].getMaxCost()
        self.budget = np.random.rand() * maxCost * 0.2

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
