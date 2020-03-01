from game import Game
from approxMILP import ApproxMILP
from approxMILPBS import ApproxMILPBS
from learn import LearnNN
import time
import numpy as np
import matplotlib.pyplot as plt
import copy

if __name__ == '__main__':
    seed = 99
    K = 12
    L = 100

    eps = 0.0001

    numExp = 20
    nList = np.concatenate([np.linspace(2,10,5,dtype=int), [20], np.linspace(50, 200, 4,dtype=int)])

    numN = nList.size

    meanResults = -1 * np.ones((numN,12))
    stdResults = -1 * np.ones((numN,12))
    for ni in range(numN):
        n = int(nList[ni])
        nResults = -1 * np.ones((numExp, 12))
        for seed in np.arange(numExp):
            print('n = ', n, ', seed = ', seed)
            trueG = Game(seed, n, K)

            learnModel = LearnNN(trueG)
            CF_model = learnModel.learnCF()
            GD_model = learnModel.learnGD()



            # Test closed form learning
            planG = copy.deepcopy(trueG)
            planG.setModel(CF_model)

            planModel = ApproxMILP(planG, int(L))
            if n <= 20:
                planModel.solve()
                planModel.evaluateSolution(trueG)

            planModelBS = ApproxMILPBS(planG, int(L))
            planModelBS.solveBS()
            planModelBS.evaluateSolution(trueG)

            nResults[seed, 0] = planModel.trueobj
            nResults[seed, 1] = planModelBS.trueobj

            nResults[seed, 3] = np.abs(planModel.trueobj - planModel.obj) / planModel.trueobj
            nResults[seed, 4] = np.abs(planModelBS.trueobj - planModelBS.obj) / planModelBS.trueobj



            # Test gradient descent learning
            planG = copy.deepcopy(trueG)
            planG.setModel(GD_model)

            planModel = ApproxMILP(planG, int(L))
            if n <= 20:
                planModel.solve()
                planModel.evaluateSolution(trueG)

            planModelBS = ApproxMILPBS(planG, int(L))
            planModelBS.solveBS()
            planModelBS.evaluateSolution(trueG)



            nResults[seed, 6] = planModel.trueobj
            nResults[seed, 7] = planModelBS.trueobj

            nResults[seed, 9] = np.abs(planModel.trueobj - planModel.obj) / planModel.trueobj
            nResults[seed, 10] = np.abs(planModelBS.trueobj - planModelBS.obj) / planModelBS.trueobj


            truePlanModelBS = ApproxMILPBS(trueG, int(L))
            truePlanModelBS.solveBS()

            nResults[seed, 0] = (nResults[seed, 0] - truePlanModelBS.obj) / truePlanModelBS.obj
            nResults[seed, 1] = (nResults[seed, 1] - truePlanModelBS.obj) / truePlanModelBS.obj
            nResults[seed, 6] = (nResults[seed, 6] - truePlanModelBS.obj) / truePlanModelBS.obj
            nResults[seed, 7] = (nResults[seed, 7] - truePlanModelBS.obj) / truePlanModelBS.obj

            print('True Optimal: ', truePlanModelBS.obj, 'CF Optimal: ', nResults[seed, 1], 'GD Optimal: ', nResults[seed, 7])

            nResults[seed, 11] = learnModel.getCFBound()
            np.savetxt(str(n)+'Results.csv', nResults, delimiter=',')

        meanResults[ni,:] = np.mean(nResults, axis=0)
        stdResults[ni,:] = np.std(nResults, axis=0)

        np.savetxt('meanResults.csv', meanResults, delimiter=',')
        np.savetxt('stdResults.csv', stdResults, delimiter=',')



    meanResults = np.loadtxt('meanResults.csv', delimiter=',')
    stdResults = np.loadtxt('stdResults.csv', delimiter=',')

    fig, (timeplt) = plt.subplots(1, 1)
    ind = np.arange(numN)  # the x locations for the groups
    width = 0.2  # the width of the bars
    time1 = timeplt.bar(ind - 3 * width / 2, meanResults[:,0], width, yerr=stdResults[:,0],
                    color='Blue', label='MILP-CF', log=True)
    time2 = timeplt.bar(ind - width / 2, meanResults[:,1], width, yerr=stdResults[:,1],
                    color='Red', label='MILPBS-CF', log=True)
    time3 = timeplt.bar(ind + width / 2, meanResults[:,6], width, yerr=stdResults[:,6],
                    color='Yellow', label='MILP-GD', log=True)
    time4 = timeplt.bar(ind + 3 * width / 2, meanResults[:,7], width, yerr=stdResults[:,7],
                    color='Green', label='MILPBS-GD', log=True)

    timeplt.set_ylabel('Solution Gap')
    timeplt.set_xticks(ind)
    timeplt.set_xticklabels(nList)
    timeplt.set_xlabel('Number of targets')
    timeplt.set_ybound(lower=1e-6, upper=0.3)
    timeplt.legend()

    plt.show()
