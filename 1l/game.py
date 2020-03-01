from node import Node
from nn import NN
import random
import numpy as np
import torch
class Game:
    def __init__(self, seed, n, K):
        self.n = n
        self.K = K
        self.Kd = int(K*2/3)
        self.Kc = int(K*1/3)
        self.seed = seed
        np.random.seed(seed)
        self.eps = 1e-8
        self.CfeatureWeights = np.random.rand(self.Kc)-0.5
        self.DfeatureWeights = np.random.rand(self.Kd)-0.5
        for k in range(self.Kc):
            if self.CfeatureWeights[k] == 0:
                self.CfeatureWeights[k] = self.eps
        for k in range(self.Kd):
            if self.DfeatureWeights[k] == 0:
                self.DfeatureWeights[k] = self.eps
        self.nodes = [Node(K, self.CfeatureWeights, self.DfeatureWeights, seed) for i in range(n)]
        self.us = np.array([node.u for node in self.nodes])
        maxCost = 0
        for i in range(n):
            maxCost += self.nodes[i].getMaxCost()
        self.budget = np.random.rand() * maxCost * 0.2

        self.f = NN(K, 10, 20, False)
        self.allWeights = np.concatenate([self.CfeatureWeights, self.DfeatureWeights])
        self.f.input_linear.weight = torch.nn.Parameter(torch.unsqueeze(torch.tensor(self.allWeights, dtype=torch.float), dim=0))

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

    def setModel(self, model):
        self.f = model
        self.allWeights = np.array(self.f.input_linear.weight.data)
        self.allWeights = self.allWeights[0]
        for k in range(self.K):
            if self.allWeights[k] == 0:
                self.allWeights[k] = self.eps
        self.CfeatureWeights = self.allWeights[:self.Kc]
        self.DfeatureWeights = self.allWeights[self.Kc:]
