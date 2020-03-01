from game import Game
from approxMILP import ApproxMILP
from approxMILPBS import ApproxMILPBS
import time
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    seed = 99
    K = 12
    L = 100

    numExp = 20
    nList = np.concatenate([np.linspace(2,10,5,dtype=int), [20], np.linspace(50, 200, 4, dtype=int)])
    numN = nList.size

    meanResults = -1 * np.ones((numN,12))
    stdResults = -1 * np.ones((numN,12))
    for ni in range(numN):
        n = int(nList[ni])
        nResults = -1 * np.ones((numExp, 12))
        for seed in np.arange(numExp):
            print('n = ', n, ', seed = ', seed)
            g = Game(seed, n, K)

            model = ApproxMILP(g, int(L))
            if n <= 20:
                model.solve()

            modelBS = ApproxMILPBS(g, int(L))
            modelBS.solveBS()


            nResults[seed, 0] = model.solveTime
            nResults[seed, 1] = modelBS.solveTime

            nResults[seed, 3] = model.initTime
            nResults[seed, 4] = modelBS.initTime

            nResults[seed, 6] = model.obj
            nResults[seed, 7] = modelBS.obj
            optSolution = modelBS.obj

            nResults[seed, 6] = (model.obj - optSolution) / optSolution
            nResults[seed, 7] = (modelBS.obj - optSolution) / optSolution

            nResults[seed, 9] = model.bound
            nResults[seed, 10] = modelBS.bound

            np.savetxt('meanResults.csv', meanResults, delimiter=',')
            np.savetxt('stdResults.csv', stdResults, delimiter=',')

        meanResults[ni,:] = np.mean(nResults, axis=0)
        stdResults[ni,:] = np.std(nResults, axis=0)

        np.savetxt('meanResults.csv', meanResults, delimiter=',')
        np.savetxt('stdResults.csv', stdResults, delimiter=',')



    # meanResults = np.loadtxt('meanResults.csv', delimiter=',')
    # stdResults = np.loadtxt('stdResults.csv', delimiter=',')

    fig, (timeplt) = plt.subplots(1, 1)
    ind = np.arange(numN)  # the x locations for the groups
    width = 0.2  # the width of the bars
    time1 = timeplt.bar(ind - width, meanResults[:,0], width, yerr=stdResults[:,0],
                    color='Blue', label='MILP')
    time2 = timeplt.bar(ind, meanResults[:,1], width, yerr=stdResults[:,1],
                    color='Red', label='MILP-BS')

    timeplt.set_ylabel('Running Time (s)')
    timeplt.set_xticks(ind)
    timeplt.set_xticklabels(nList)
    timeplt.set_xlabel('Number of targets')
    timeplt.legend()

    plt.show()
