# -*- coding: utf-8 -*-
import torch
import numpy as np
from nn import MyLossReg
from nn import MyLoss
from nn import MyLossId
from nn import MyLossNeg
import time

class nnGreedySolve:
    def __init__(self, game):
        self.initStart = time.time()
        self.game = game
        self.n = game.n
        self.K = game.K
        self.K2 = game.K2
        self.K3 = game.K3
        self.xhat = torch.tensor(game.xhat, dtype=torch.float)
        self.xlow = torch.tensor(game.xhat, requires_grad=True, dtype=torch.float)
        self.xhigh = torch.tensor(game.xhat, requires_grad=True, dtype=torch.float)
        self.xalter = torch.tensor(game.xhat, requires_grad=True, dtype=torch.float)
        self.x1 = torch.tensor(game.xhat[0,:], requires_grad=True, dtype=torch.float)
        self.us = torch.tensor(game.us, dtype=torch.float)
        self.costs = torch.tensor(game.costs, dtype=torch.float)
        self.budget = game.budget
        self.model = game.f
        self.model.train = False

        self.regCoe = 1
        self.lr = 1e-3
        self.initTime = time.time() - self.initStart

    def getMin(self):
        criterion = MyLossId()
        optimizer = torch.optim.RMSprop([self.x1], lr=self.lr)
        for t in range(1000):
            y = self.model(self.x1)
            y = y.squeeze()
            self.loss = criterion(y)
            optimizer.zero_grad()
            self.loss.backward()
            optimizer.step()
            self.x1.data.clamp_(min=0, max=1)
        self.xmin = torch.tensor(self.x1)
        self.minf = self.loss.data

    def getMax(self):
        criterion = MyLossNeg()
        optimizer = torch.optim.RMSprop([self.x1], lr=self.lr)
        for t in range(1000):
            y = self.model(self.x1)
            y = y.squeeze()
            self.loss = criterion(y)
            optimizer.zero_grad()
            self.loss.backward()
            optimizer.step()
            self.x1.data.clamp_(min=0, max=1)
        self.xmax = torch.tensor(self.x1)
        self.maxf = -self.loss.data

    def getCost(self, xtarget, xhat, cost):
        diff = xtarget - xhat
        return np.dot(diff.abs().flatten(), cost.flatten())

    def lowSol(self):
        budgetLeft = self.budget
        for i in range(self.n):
            thisCost = self.getCost(self.xmax, self.xhat[self.uorder[i],:], self.costs[self.uorder[i],:])
            if thisCost <= budgetLeft:
                self.xlow[self.uorder[i],:] = self.xmax
                budgetLeft -= thisCost
        self.lowloss = self.evaluatePoint(self.xlow)
        print("Final low value: ", self.lowloss)
        return self.lowloss.data

    def highSol(self):
        budgetLeft = self.budget
        for i in range(self.n-1, -1, -1):
            thisCost = self.getCost(self.xmin, self.xhat[self.uorder[i],:], self.costs[self.uorder[i],:])
            if thisCost <= budgetLeft:
                self.xhigh[self.uorder[i],:] = self.xmin
                budgetLeft -= thisCost
        self.highloss = self.evaluatePoint(self.xhigh)
        print("Final high value: ", self.highloss)
        return self.highloss.data


    def alterSol(self):
        budgetLeft = self.budget
        lowIndex = 0
        highIndex = self.n-1
        moved = True
        while budgetLeft > 0 and lowIndex < highIndex and moved:
            moved = False
            lowCost = self.getCost(self.xmax, self.xhat[self.uorder[lowIndex],:],
             self.costs[self.uorder[lowIndex],:])
            if lowCost <= budgetLeft:
                self.xalter[self.uorder[lowIndex],:] = self.xmax
                budgetLeft -= lowCost
                lowIndex += 1
                moved = True
            highCost = self.getCost(self.xmin, self.xhat[self.uorder[highIndex],:],
             self.costs[self.uorder[highIndex],:])
            if highCost <= budgetLeft:
                self.xalter[self.uorder[highIndex],:] = self.xmin
                budgetLeft -= highCost
                highIndex -= 1
                moved = True
        self.alterloss = self.evaluatePoint(self.xalter)
        return self.alterloss.data


    def solve(self):
        self.solveStart = time.time()
        self.getMin()
        self.x1 = torch.tensor(self.game.xhat[0,:], requires_grad=True, dtype=torch.float)
        self.getMax()
        self.uorder = np.argsort(self.us)
        self.alterSol()
        self.solveTime = time.time() - self.solveStart
        self.optVal = self.alterloss.item()
        return self.optVal




    def evaluatePoint(self, sol):
        criterion = MyLossReg(self.regCoe, self.costs, self.xhat, self.budget)
        sol = torch.tensor(sol, dtype=torch.float)
        y = self.model(sol)
        y = y.squeeze()
        self.loss, self.actualCost = criterion(y, self.us, sol)
        return self.loss


    def evaluateSolution(self, trueGame):
        true_model = trueGame.f
        criterion = MyLossReg(self.regCoe, self.costs, self.xhat, self.budget)
        y = true_model(self.xalter)
        y = y.squeeze()
        self.trueoptVal, self.trueActualCost = criterion(y, self.us, self.xalter)
        self.trueoptVal = self.trueoptVal.item()
        return self.trueoptVal
