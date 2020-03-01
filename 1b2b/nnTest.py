from nnGame import Game
from nnIpoptSolve import nnIpoptSolve
from nnSolve import nnSolve
from nnGreedySolve import nnGreedySolve
import time
import numpy as np

if __name__ == '__main__':
    seed = 99
    n = 20
    K = 20
    K2 = 30
    K3 = 50
    g = Game(seed, n, K, K2, K3)
    modelIpopt = nnIpoptSolve(g)
    modelIpopt.solve()
