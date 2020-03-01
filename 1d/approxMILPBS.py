from game import Game
from docplex.mp.model import Model
from docplex.mp.solution import SolveSolution
import numpy as np
import time
class ApproxMILPBS:
    def __init__(self, game, L):
        self.initStart = time.time()
        self.game = game
        self.nodes = game.nodes
        self.n = game.n
        self.K = game.K
        self.Kd = game.Kd
        self.Kc = game.Kc
        self.model = Model(name='approxMILP')
        self.W = np.sum(np.abs(game.CfeatureWeights)) + np.sum(np.abs(game.DfeatureWeights))
        self.L = L
        self.scores = None


    def solveBS(self):
        self.ub = 1.0
        self.lb = 0
        self.interval = self.ub - self.lb
        self.delta = (self.ub + self.lb) / 2
        iter = 0
        self.prep()
        self.generateMILP()
        self.initTime = time.time() - self.initStart
        self.solveStart = time.time()
        while self.interval > 0.0001 and iter < 30:
            self.solve()
            if self.optVal < 0:
                self.ub = self.delta
            elif self.optVal > 0:
                self.lb = self.delta
            else:
                print("Unsafe branch, terminate")
                break
            self.interval = self.ub - self.lb
            self.delta = (self.ub + self.lb) / 2
            self.updateObjective()
            iter += 1
        self.solveTime = time.time() - self.solveStart
        self.bound = self.eps * self.eps * 2.0 + self.interval
        print("OBJ value Compute = ", self.obj)
        return self.n, self.K, self.eps, self.obj, self.eps * self.eps * 2.0, self.interval


    def updateObjective(self):
        us = [node.u for node in self.nodes]
        self.model.minimize(self.model.scal_prod(list(self.f.values()), us) - self.delta * self.model.sum(self.f))


    def prep(self):
        self.grid = np.linspace(-2*self.W, 0, self.L+1)
        self.eps = self.grid[1] - self.grid[0]
        self.expValue = np.exp(self.grid)
        self.gamma = (self.expValue[1:] - self.expValue[:-1]) / (self.grid[1:] - self.grid[:-1])
        self.gamma = np.flip(self.gamma)

    def generateMILP(self):
        us = [node.u for node in self.nodes]
        M = 1/(np.exp(-2*self.W) * sum(us))
        self.y = self.model.binary_var_matrix(self.n, self.L, name="y")
        self.xd = self.model.binary_var_matrix(self.n, self.Kd, name="xd")
        self.xc = self.model.continuous_var_matrix(self.n, self.Kc, name="xc")
        self.z = self.model.continuous_var_matrix(self.n, self.L, ub=self.eps, name="z")
        self.h = self.model.continuous_var_matrix(self.n, self.Kc, name="h")
        self.f = self.model.continuous_var_dict(self.n, name="f")



        self.sumDisCost = self.model.linear_expr()
        for i in range(self.n):
            for k in range(self.Kd):
                self.sumDisCost = self.sumDisCost + self.nodes[i].Dfeatures[k].cost * self.xd[(i,k)]

        self.sumConCost = self.model.linear_expr()
        for i in range(self.n):
            expry = self.model.linear_expr()
            for k in range(self.Kc):
                expr = self.model.linear_expr(self.h[(i,k)] - self.xc[(i,k)] + self.nodes[i].Cfeatures[k].xhat)
                self.model.add_constraint(expr >= 0)
                expr = self.model.linear_expr(self.h[(i,k)] + self.xc[(i,k)] - self.nodes[i].Cfeatures[k].xhat)
                self.model.add_constraint(expr >= 0)
                expr = self.model.linear_expr(self.xc[(i,k)] - max(self.nodes[i].Cfeatures[k].xhat -
                self.nodes[i].Cfeatures[k].range, 0))
                self.model.add_constraint(expr >= 0)
                expr = self.model.linear_expr(self.xc[(i,k)] - min(self.nodes[i].Cfeatures[k].xhat +
                self.nodes[i].Cfeatures[k].range, 1))
                self.model.add_constraint(expr <= 0)
                self.sumConCost = self.sumConCost + self.nodes[i].Cfeatures[k].cost * self.h[(i,k)]


        for i in range(self.n):
            epsz = [ self.model.linear_expr(self.eps - self.z[(i, l)]) for l in range(self.L)]
            self.model.add_constraint( self.model.linear_expr(
                self.f[i] - np.exp(-2*self.W) - self.model.scal_prod(epsz, self.gamma)) == 0)
            wqC = [self.model.linear_expr(self.game.CfeatureWeights[k] * self.xc[(i, k)]) for k in range(self.Kc)]
            wbD = [self.model.linear_expr(self.game.DfeatureWeights[k] * self.xd[(i, k)]) for k in range(self.Kd)]
            zz = [self.z[(i, l)] for l in range(self.L)]
            self.model.add_constraint( self.model.linear_expr(
                self.model.sum(zz) + self.model.sum(wqC) + self.model.sum(wbD) - self.W) == 0)

            for l in range(self.L):
                expr = self.model.linear_expr(self.eps * self.y[(i,l)] - self.z[(i,l)])
                self.model.add_constraint(expr <= 0)
                if l != self.L-1:
                    expr = self.model.linear_expr(self.eps * self.y[(i,l)] - self.z[(i,l+1)])
                    self.model.add_constraint(expr >= 0)

        group = np.arange(self.Kd).reshape((2,-1))
        for i in range(self.n):
            for j in range(group.shape[0]):
                sumxd = self.model.linear_expr()
                for k in group[j,:]:
                    sumxd = sumxd + self.xd[(i,k)]
                self.model.add_constraint(sumxd == 1)


        self.model.add_constraint( self.sumDisCost + self.sumConCost  <= self.game.budget)



        self.model.minimize(self.model.scal_prod(list(self.f.values()), us) - self.delta * self.model.sum(self.f))

    def solve(self):
        self.model.parameters.mip.tolerances.mipgap = 1e-9
        solution = self.model.solve()
        self.model.export_as_lp("discMILP")
        if not solution:
            self.feasible = False
        else:
            self.feasible = True
            solution.export("Sol")
            self.extractSolution(solution)
            return self.obj



    def extractSolution(self, solution):
        self.optVal = solution.get_objective_value()
        self.fdiff = np.zeros(self.n)
        self.xcsol = solution.get_value_dict(self.xc)
        self.xdsol = solution.get_value_dict(self.xd)
        self.fsol = solution.get_value_dict(self.f)
        self.zsol = solution.get_value_dict(self.z)

        ewx = 0
        ewxu = 0
        feasible = True
        Dcost = 0
        Ccost = 0
        epsilon = 0.0001
        for i in range(self.n):
            wx = 0
            zcount = 0
            for l in range(self.L):
                if self.zsol[(i, l)] > 0 and self.zsol[(i,l)] < self.eps:
                    zcount += 1
            for k in range(self.Kc):
                wx = wx + self.game.CfeatureWeights[k] * self.xcsol[(i, k)]
                Ccost += abs(self.xcsol[(i, k)] - self.nodes[i].Cfeatures[k].xhat) * self.nodes[i].Cfeatures[k].cost
                if abs(self.xcsol[(i, k)] - self.nodes[i].Cfeatures[k].xhat) > self.nodes[i].Cfeatures[k].range + epsilon:
                    feasible = False
                    print("Infisible x out of bound, ", i, k, abs(self.xcsol[(i, k)] - self.nodes[i].Cfeatures[k].xhat), self.nodes[i].Cfeatures[k].range)
            for k in range(self.Kd):
                wx = wx + self.game.DfeatureWeights[k] * self.xdsol[(i, k)]
                Dcost += self.xdsol[(i, k)] * self.nodes[i].Dfeatures[k].cost
            self.fdiff[i] = self.fsol[i] - np.exp(wx - self.W)
            if self.fdiff[i] > self.eps * self.eps / 2.0:
                print("FERROR")
            ewx += np.exp(wx - self.W)
            ewxu += np.exp(wx - self.W) * self.nodes[i].u
        self.obj = ewxu / ewx
        if Dcost + Ccost > self.game.budget + epsilon:
            print("Infisible cost, discrete Cost = ", Dcost, ", continuous cost = ", Ccost, self.game.budget)
        us = np.array([node.u for node in self.nodes])
        dus = us - self.delta
        dus[dus<0] = 0
        dus = dus * (self.eps * self.eps / 2)
        self.threshold = np.sum(dus) + np.abs(np.dot(dus, self.fdiff))


    def getScores(self):
        return self.xsol
