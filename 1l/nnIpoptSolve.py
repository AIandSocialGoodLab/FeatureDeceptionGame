# from __future__ import division
from pyomo.environ import *
from pyomo.opt import SolverStatus, TerminationCondition
import numpy as np
import time
from nn import MyLossReg
import torch

class nnIpoptSolve:
    def __init__(self, game):
        self.initStart = time.time()
        self.game = game
        self.n = game.n
        self.K = game.K
        self.xhat = game.xhat
        self.us = game.us
        self.costs = game.costs
        self.budget = game.budget
        self.f = game.f
        self.layers = len(game.f.input_linear)
        self.weights = list()
        for i in range(self.layers):
            self.weights.append(np.array(game.f.input_linear[i].weight.data))
        self.bias = list()
        for i in range(self.layers-1):
            self.bias.append(np.array(game.f.input_linear[i].bias.data))
        self.generateNLP()
        self.f.train = False
        self.initTime = time.time() - self.initStart

    def generateNLP(self):
        self.model = ConcreteModel()
        self.model.xSet = Set(initialize=range(self.n)) * Set(initialize=range(self.K))
        self.model.nSet = Set(initialize=range(self.n))
        self.model.kSet = Set(initialize=range(self.K))

        def xb(model, i, k):
            return (0, 1)
        self.model.x = Var(self.model.nSet, self.model.kSet, domain=NonNegativeReals, bounds=xb)

        for i in range(self.n):
            for k in range(self.K):
                self.model.x[i,k].value = self.xhat[i,k]


        score = list()
        for i in range(self.n):
            lin = list()
            for kk in range(self.K):
                lin.append(self.model.x[i,kk])
            for l in range(self.layers):
                W = self.weights[l]

                lout = list()
                if l != 0:
                    b = self.bias[l-1]
                    for ii in range(len(lin)):
                        lin[ii] = tanh(lin[ii] + b[ii])
                for k in range(W.shape[0]):
                    lout.append(sum(W[k,j] * lin[j] for j in range(W.shape[1])))
                lin = lout.copy()
            score.append(lout[0])
        exprNum = sum(exp(score[i]) * self.us[i] for i in range(self.n))
        exprDen = sum(exp(score[i]) for i in range(self.n))


        expr = exprNum / exprDen
        self.model.obj = Objective(expr=expr, sense=minimize)
        self.model.h = Var(self.model.nSet, self.model.kSet, domain=NonNegativeReals)
        self.model.absConstraint = ConstraintList()
        for i in range(self.n):
            for k in range(self.K):
                self.model.absConstraint.add(abs(self.model.x[i,k] - self.xhat[i,k]) == self.model.h[i,k])
        self.model.budgetConstraint = Constraint(expr = sum(self.model.h[i,k] * self.costs[i,k] for i in range(self.n) for k in range(self.K)) <= self.budget)


    def solve(self):
        self.solveStart = time.time()
        solver = pyomo.opt.SolverFactory('ipopt')
        solver.options['print_level'] = 0
        solver.options['max_iter'] = int(100)
        solver.options['max_cpu_time'] = int(60)
        solver.options['warm_start_init_point'] = 'yes'
        result = solver.solve(self.model, tee = True)
        if (result.solver.status == SolverStatus.ok) and (result.solver.termination_condition == TerminationCondition.optimal):
            self.feasible = True
            self.extractSolution(result)
            self.solveTime = time.time() - self.solveStart
            return self.optVal, self.xsol
        else:
            self.feasible = False
            self.extractSolution(result)
            self.solveTime = time.time() - self.solveStart
            return self.optVal, self.xsol

    def extractSolution(self, solution):
        self.optVal = value(self.model.obj)
        self.xsol = [[value(self.model.x[i,k]) for k in range(self.K)] for i in range(self.n)]
        s = 0
        for i in range(self.n):
            for j in range(self.K):
                s = s + self.costs[i,j] * abs(self.xsol[i][j] - self.xhat[i,j])
    def evaluateSolution(self, trueGame):
        true_model = trueGame.f
        self.regCoe = 1
        criterion = MyLossReg(self.regCoe, torch.tensor(self.costs, dtype=torch.float), torch.tensor(self.xhat, dtype=torch.float)
        , torch.tensor(self.budget, dtype=torch.float))
        self.xsol = torch.tensor(self.xsol, dtype=torch.float)
        y = true_model(self.xsol)
        y = y.squeeze()
        self.trueoptVal, self.trueActualCost = criterion(y, torch.tensor(self.us, dtype=torch.float), self.xsol)
        self.trueoptVal = self.trueoptVal.item()

        y = true_model(torch.tensor(self.xhat, dtype=torch.float))
        y = y.squeeze()
        self.trueoriVal, self.trueoriCost = criterion(y, torch.tensor(self.us, dtype=torch.float), torch.tensor(self.xhat, dtype=torch.float))
        self.trueoriVal = self.trueoriVal.item()
        print('ipopt optimized loss: ', self.trueoptVal, ', initial loss: ', self.trueoriVal)

        return self.trueoptVal
