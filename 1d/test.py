from game import Game
from approxMILP import ApproxMILP
from approxMILPBS import ApproxMILPBS
import time
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    seed = 99
    n = 10
    K = 12
    L = 100

    numExp = 20
    kList = np.linspace(6,30,5,dtype=int)
    numK = kList.size

    meakResults = -1 * np.ones((numK,12))
    stdResults = -1 * np.ones((numK,12))
    for ki in range(numK):
        K = int(kList[ki])
        kResults = -1 * np.ones((numExp, 12))
        for seed in np.arange(numExp):
            print('K = ', K, ', seed = ', seed)
            g = Game(seed, n, K)

            model = ApproxMILP(g, int(L))
            if K <= 20:
                model.solve()

            modelBS = ApproxMILPBS(g, int(L))
            modelBS.solveBS()


            kResults[seed, 0] = model.solveTime
            kResults[seed, 1] = modelBS.solveTime

            kResults[seed, 3] = model.initTime
            kResults[seed, 4] = modelBS.initTime

            kResults[seed, 6] = model.obj
            kResults[seed, 7] = modelBS.obj

            optSolution = min(kResults[seed, 6:8])

            kResults[seed, 6] = (model.obj - optSolution) / optSolution
            kResults[seed, 7] = (modelBS.obj - optSolution) / optSolution

            kResults[seed, 9] = model.bound
            kResults[seed, 10] = modelBS.bound

            np.savetxt('meakResults.csv', meakResults, delimiter=',')
            np.savetxt('stdResults.csv', stdResults, delimiter=',')

        meakResults[ki,:] = np.mean(kResults, axis=0)
        stdResults[ki,:] = np.std(kResults, axis=0)

        np.savetxt('meakResults.csv', meakResults, delimiter=',')
        np.savetxt('stdResults.csv', stdResults, delimiter=',')



    meakResults = np.loadtxt('meakResults.csv', delimiter=',')
    stdResults = np.loadtxt('stdResults.csv', delimiter=',')

    fig, (timeplt) = plt.subplots(1, 1)
    ind = np.arange(numK)  # the x locations for the groups
    width = 0.35  # the width of the bars
    time1 = timeplt.bar(ind - width/2, meakResults[:,0], width, yerr=stdResults[:,0],
                    color='Blue', label='MILP', log=True)
    time2 = timeplt.bar(ind + width/2, meakResults[:,1], width, yerr=stdResults[:,1],
                    color='Red', label='MILP-BS', log=True)
    timeplt.set_ylabel('Running Time (s)')
    timeplt.set_xticks(ind)
    timeplt.set_xticklabels(kList)
    timeplt.set_xlabel('Features')
    timeplt.legend()

    plt.show()
