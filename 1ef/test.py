from nnGame import Game
from nnIpoptSolve import nnIpoptSolve
from nnSolve import nnSolve
from nnGreedySolve import nnGreedySolve
import time
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    seed = 99
    n = 5
    K = 12
    K2 = 24
    K3 = 12
    numExp = 20
    numN = 6
    nstep = 5
    nList = np.linspace(5, 30, 6, dtype=int)
    meanResults = -1 * np.ones((numN,9))
    stdResults = -1 * np.ones((numN,9))
    headertxt = "seed, IpoptVal, IpoptTime, NNVal, NNTime"
    for ni in range(numN):
        n = nList[ni]
        nResults = -1 * np.ones((numExp, 9))
        for seed in np.arange(numExp):
            print('n = ', n, ', seed = ', seed)
            g = Game(seed, n, K, K2, K3)

            modelIpopt = nnIpoptSolve(g)
            modelIpopt.solve()


            modelNN = nnSolve(g)
            modelNN.solve()

            modelGreedy = nnGreedySolve(g)
            modelGreedy.solve()

            nResults[seed, 0] = modelIpopt.solveTime
            nResults[seed, 1] = modelNN.solveTime
            nResults[seed, 2] = modelGreedy.solveTime

            nResults[seed, 3] = modelIpopt.initTime
            nResults[seed, 4] = modelNN.initTime
            nResults[seed, 5] = modelGreedy.initTime

            nResults[seed, 6] = modelIpopt.optVal
            nResults[seed, 7] = modelNN.optVal
            nResults[seed, 8] = modelGreedy.optVal

            optSolution = modelNN.optVal

            nResults[seed, 6] = (modelIpopt.optVal - optSolution) / optSolution
            nResults[seed, 7] = (modelNN.optVal - optSolution) / optSolution
            nResults[seed, 8] = (modelGreedy.optVal - optSolution) / optSolution

            np.savetxt('meanResults.csv', meanResults, delimiter=',')
            np.savetxt('stdResults.csv', stdResults, delimiter=',')

        meanResults[ni,:] = np.mean(nResults, axis=0)
        stdResults[ni,:] = np.std(nResults, axis=0)

        np.savetxt('meanResults.csv', meanResults, delimiter=',')
        np.savetxt('stdResults.csv', stdResults, delimiter=',')


    meanResults = np.loadtxt('meanResults.csv', delimiter=',')
    stdResults = np.loadtxt('stdResults.csv', delimiter=',')

    fig, (timeplt, errorplt) = plt.subplots(1, 2)
    ind = np.arange(numN)  # the x locations for the groups
    width = 0.2  # the width of the bars
    time1 = timeplt.bar(ind - width, meanResults[:,0], width, yerr=stdResults[:,0],
                    color='SkyBlue', label='IPOPT')
    time2 = timeplt.bar(ind, meanResults[:,1], width, yerr=stdResults[:,1],
                    color='Red', label='Gradient Descent')
    time3 = timeplt.bar(ind + width, meanResults[:,2], width, yerr=stdResults[:,2],
                    color='Brown', label='Greedy')

    error1 = errorplt.bar(ind - width, meanResults[:,6], width, yerr=stdResults[:,6],
                    color='SkyBlue', label='IPOPT')
    error2 = errorplt.bar(ind, meanResults[:,7], width, yerr=stdResults[:,7],
                    color='Red', label='Gradient Descent')
    error3 = errorplt.bar(ind + width, meanResults[:,8], width, yerr=stdResults[:,8],
                    color='Brown', label='Greedy')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    timeplt.set_ylabel('Running Time (s)')
    timeplt.set_xticks(ind)
    timeplt.set_xticklabels(nList)
    timeplt.set_xlabel('Number of targets')
    timeplt.legend()

    errorplt.set_ylabel('Solution Gap')
    errorplt.set_xticks(ind)
    errorplt.set_xticklabels(nList)
    errorplt.set_xlabel('Number of targets')
    errorplt.legend()

    plt.show()
