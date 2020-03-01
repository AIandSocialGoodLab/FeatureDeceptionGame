from game import Game
from docplex.mp.model import Model
from docplex.mp.solution import SolveSolution
import numpy as np
import time
class ApproxMILP:
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
        self.prep()
        self.generateMILP()
        self.initTime = time.time() - self.initStart
        self.solveTime = 0
        self.obj = 0

    def prep(self):
        self.grid = np.linspace(-2*self.W, 0, self.L+1)
        self.eps = self.grid[1] - self.grid[0]
        self.expValue = np.exp(self.grid)
        self.gamma = (self.expValue[1:] - self.expValue[:-1]) / (self.grid[1:] - self.grid[:-1])
        self.gamma = np.flip(self.gamma)
        self.bound = self.eps * self.eps * 2.0


    def generateMILP(self):
        us = [node.u for node in self.nodes]
        M = 1/(np.exp(-2*self.W) * sum(us))
        self.y = self.model.binary_var_matrix(self.n, self.L, name="y")
        self.d = self.model.binary_var_matrix(self.n, self.Kd, name="d")
        self.v = self.model.continuous_var(name="v")
        self.q = self.model.continuous_var_matrix(self.n, self.Kc, name="q")
        self.s = self.model.continuous_var_matrix(self.n, self.L, name="s")
        self.h = self.model.continuous_var_matrix(self.n, self.Kc, name="h")
        self.g = self.model.continuous_var_matrix(self.n, self.L, name="g")
        self.b = self.model.continuous_var_matrix(self.n, self.Kd, name="b")
        self.t = self.model.continuous_var_dict(self.n, name="t")

        for i in range(self.n):
            for l in range(self.L):
                self.model.add_constraint(self.g[(i,l)] <= self.v)
                self.model.add_constraint(self.g[(i,l)] <= M * self.y[(i,l)])
                self.model.add_constraint(self.g[(i,l)] >= self.v - M * (1 - self.y[(i,l)]))

        self.sumDisCost = self.model.linear_expr()
        for i in range(self.n):
            for k in range(self.Kd):
                self.model.add_constraint(self.b[(i,k)] <= self.v)
                self.model.add_constraint(self.b[(i,k)] <= M * self.d[(i,k)])
                self.model.add_constraint(self.b[(i,k)] >= self.v - M * (1 - self.d[(i,k)]))
                self.sumDisCost = self.sumDisCost + self.nodes[i].Dfeatures[k].cost * self.b[(i,k)]

        self.sumConCost = self.model.linear_expr()
        for i in range(self.n):
            expry = self.model.linear_expr()
            for k in range(self.Kc):
                expr = self.model.linear_expr(self.h[(i,k)] - self.q[(i,k)] + self.nodes[i].Cfeatures[k].xhat * self.v)
                self.model.add_constraint(expr >= 0)
                expr = self.model.linear_expr(self.h[(i,k)] + self.q[(i,k)] - self.nodes[i].Cfeatures[k].xhat * self.v)
                self.model.add_constraint(expr >= 0)
                expr = self.model.linear_expr(self.q[(i,k)] - max(self.nodes[i].Cfeatures[k].xhat -
                self.nodes[i].Cfeatures[k].range, 0) * self.v)
                self.model.add_constraint(expr >= 0)
                expr = self.model.linear_expr(self.q[(i,k)] - min(self.nodes[i].Cfeatures[k].xhat +
                self.nodes[i].Cfeatures[k].range, 1) * self.v)
                self.model.add_constraint(expr <= 0)
                self.sumConCost = self.sumConCost + self.nodes[i].Cfeatures[k].cost * self.h[(i,k)]


        for i in range(self.n):
            vepss = [ self.model.linear_expr(self.v * self.eps - self.s[(i, l)]) for l in range(self.L)]
            self.model.add_constraint( self.model.linear_expr(
                self.t[i] - np.exp(-2*self.W) * self.v - self.model.scal_prod(vepss, self.gamma)) == 0)
            wqC = [self.model.linear_expr(self.game.CfeatureWeights[k] * self.q[(i, k)]) for k in range(self.Kc)]
            wbD = [self.model.linear_expr(self.game.DfeatureWeights[k] * self.b[(i, k)]) for k in range(self.Kd)]
            ss = [self.s[(i, l)] for l in range(self.L)]
            self.model.add_constraint( self.model.linear_expr(
                self.model.sum(ss) + self.model.sum(wqC) + self.model.sum(wbD) - self.W * self.v) == 0)

            for l in range(self.L):
                expr = self.model.linear_expr(self.eps * self.g[(i,l)] - self.s[(i,l)])
                self.model.add_constraint(expr <= 0)
                if l != self.L-1:
                    expr = self.model.linear_expr(self.eps * self.g[(i,l)] - self.s[(i,l+1)])
                    self.model.add_constraint(expr >= 0)
                expr = self.model.linear_expr(self.s[(i,l)] - self.eps * self.v)
                self.model.add_constraint(expr <= 0)
        group = np.arange(self.Kd).reshape((4,-1))
        for i in range(self.n):
            for j in range(group.shape[0]):
                sumxd = self.model.linear_expr()
                for k in group[j,:]:
                    sumxd = sumxd + self.d[(i,k)]
                self.model.add_constraint(sumxd == 1)


        self.model.add_constraint( self.sumDisCost + self.sumConCost  <= self.game.budget * self.v)
        self.model.add_constraint( self.model.scal_prod(list(self.t.values()), us) == 1)



        self.model.maximize(self.model.sum(self.t))

    def solve(self):
        self.solveStart = time.time()
        self.model.parameters.mip.tolerances.mipgap = 1e-9
        self.model.set_time_limit(100)
        solution = self.model.solve()
        self.model.export_as_lp("discMILP")
        if not solution:
            self.feasible = False
            self.solveTime = time.time() - self.solveStart
            self.obj = None
        else:
            self.feasible = True
            solution.export("Sol")
            self.extractSolution(solution)
            self.solveTime = time.time() - self.solveStart
            print("OBJ value Compute = ", self.obj)



    def extractSolution(self, solution):
        self.optVal = solution.get_objective_value()
        qsol = solution.get_value_dict(self.q)
        tsol = solution.get_value_dict(self.t)
        vsol = solution.get_value(self.v)
        ssol = solution.get_value_dict(self.s)
        bsol = solution.get_value_dict(self.b)
        gsol = solution.get_value_dict(self.g)
        yysol = solution.get_value_dict(self.y)
        self.xsol = {(i, k): qsol[(i,k)] / vsol for i in range(self.n) for k in range(self.Kc)}
        self.dsol = {(i, k): bsol[(i,k)] / vsol for i in range(self.n) for k in range(self.Kd)}
        self.zsol = {(i, l): ssol[(i,l)] / vsol for i in range(self.n) for l in range(self.L)}
        self.ysol = {(i, l): gsol[(i,l)] / vsol for i in range(self.n) for l in range(self.L)}

        ewx = 0
        ewxu = 0
        feasible = True
        Dcost = 0
        Ccost = 0
        epsilon = 0.0001
        for i in range(self.n):
            wx = 0
            for k in range(self.Kc):
                wx = wx + self.game.CfeatureWeights[k] * self.xsol[(i, k)]
                Ccost += abs(self.xsol[(i, k)] - self.nodes[i].Cfeatures[k].xhat) * self.nodes[i].Cfeatures[k].cost
                if abs(self.xsol[(i, k)] - self.nodes[i].Cfeatures[k].xhat) > self.nodes[i].Cfeatures[k].range + epsilon:
                    feasible = False
                    print("Infisible x out of bound, ", i, k, abs(self.xsol[(i, k)] - self.nodes[i].Cfeatures[k].xhat), self.nodes[i].Cfeatures[k].range)
            for k in range(self.Kd):
                wx = wx + self.game.DfeatureWeights[k] * self.dsol[(i, k)]
                Dcost += self.dsol[(i, k)] * self.nodes[i].Dfeatures[k].cost
            ewx += np.exp(wx)
            ewxu += np.exp(wx) * self.nodes[i].u
        self.obj = ewxu / ewx
        if Dcost + Ccost > self.game.budget + epsilon:
            print("Infisible cost, discrete Cost = ", Dcost, ", continuous cost = ", Ccost, self.game.budget)



        z = np.zeros(self.n)
        f = np.zeros(self.n)
        for i in range(self.n):
            z[i] = 0
            for l in range(self.L):
                z[i] -= self.zsol[(i,l)]
                f[i] += self.gamma[l] * (self.eps - self.zsol[(i,l)])
            f[i] += np.exp(-2*self.W)

    def getScores(self):
        return self.xsol
