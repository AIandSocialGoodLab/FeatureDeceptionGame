# -*- coding: utf-8 -*-
import torch
import numpy as np
from nn import MyLossReg
from nn import MyLoss

class nnSolve:
    def __init__(self, game):
        self.game = game
        self.n = game.n
        self.K = game.K
        self.K2 = game.K2
        self.K3 = game.K3
        self.xhat = torch.tensor(game.xhat, dtype=torch.float)
        self.x = torch.tensor(game.xhat, requires_grad=True, dtype=torch.float)
        self.us = torch.tensor(game.us, dtype=torch.float)
        self.costs = torch.tensor(game.costs, dtype=torch.float)
        self.budget = torch.tensor(game.budget, dtype=torch.float)
        self.model = game.f

        self.regCoe = 1
        self.lr = 1e-3

    def solve(self):
        criterion = MyLossReg(self.regCoe, self.costs, self.xhat, self.budget)
        optimizer = torch.optim.RMSprop([self.x], lr=self.lr)
        for t in range(10000):
            y = self.model(self.x)
            y = y.squeeze()
            self.loss, self.actualCost = criterion(y, self.us, self.x)
            if t % 1000 == 0:
                print("t = ", t, self.loss.item(), self.actualCost, self.budget)
            optimizer.zero_grad()
            self.loss.backward()
            optimizer.step()
            self.x.data.clamp_(min=0, max=1)
        self.extractSolution()
        return self.loss.data


    def extractSolution(self):
        self.xsol = torch.tensor(self.x)
        print("OBJ value = ", self.loss)
        s = 0
        for i in range(self.n):
            for j in range(self.K):
                s = s + self.costs[i,j] * abs(self.xsol[i,j] - self.xhat[i,j])
        print("total change = ", s, "budget = ", self.budget)

    def evaluatePoint(self, sol):
        criterion = MyLossReg(self.regCoe, self.costs, self.xhat, self.budget)
        sol = torch.tensor(sol, dtype=torch.float)
        y = self.model(sol)
        y = y.squeeze()
        self.loss, self.actualCost = criterion(y, self.us, self.x)
        return self.loss
