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
    n = 5
    K = 12
    L = 100
    eps = 0.0001

    numExp = 20

    dList = np.linspace(120, 2280, 10, dtype=int)
    numD = dList.size

    meadResults = -1 * np.ones((numD,12))
    stdResults = -1 * np.ones((numD,12))
    for di in range(numD):
        numData = int(dList[di])
        dResults = -1 * np.ones((numExp, 12))
        for seed in np.arange(numExp):
            print('numData = ', numData, ', seed = ', seed)
            trueG = Game(seed, n, K)

            learnModel = LearnNN(trueG, numData)
            GD_model, CF_model = learnModel.learn()



            # Test closed form learning
            planG = copy.deepcopy(trueG)
            planG.setModel(CF_model)

            planModel = ApproxMILP(planG, int(L))
            planModel.solve()
            planModel.evaluateSolution(trueG)

            planModelBS = ApproxMILPBS(planG, int(L))
            planModelBS.solveBS()
            planModelBS.evaluateSolution(trueG)

            dResults[seed, 0] = planModel.trueobj
            dResults[seed, 1] = planModelBS.trueobj

            dResults[seed, 3] = np.abs(planModel.trueobj - planModel.obj) / planModel.trueobj
            dResults[seed, 4] = np.abs(planModelBS.trueobj - planModelBS.obj) / planModelBS.trueobj



            # Test gradient descent learning
            planG = copy.deepcopy(trueG)
            planG.setModel(GD_model)

            planModel = ApproxMILP(planG, int(L))
            planModel.solve()
            planModel.evaluateSolution(trueG)

            planModelBS = ApproxMILPBS(planG, int(L))
            planModelBS.solveBS()
            planModelBS.evaluateSolution(trueG)



            dResults[seed, 6] = planModel.trueobj
            dResults[seed, 7] = planModelBS.trueobj

            dResults[seed, 9] = np.abs(planModel.trueobj - planModel.obj) / planModel.trueobj
            dResults[seed, 10] = np.abs(planModelBS.trueobj - planModelBS.obj) / planModelBS.trueobj
            truePlanModelBS = ApproxMILPBS(trueG, int(L))
            truePlanModelBS.solveBS()

            dResults[seed, 0] = (dResults[seed, 0] - truePlanModelBS.obj) / truePlanModelBS.obj
            dResults[seed, 1] = (dResults[seed, 1] - truePlanModelBS.obj) / truePlanModelBS.obj
            dResults[seed, 6] = (dResults[seed, 6] - truePlanModelBS.obj) / truePlanModelBS.obj
            dResults[seed, 7] = (dResults[seed, 7] - truePlanModelBS.obj) / truePlanModelBS.obj

            print('True Optimal: ', truePlanModelBS.obj, 'CF Optimal: ', dResults[seed, 1], 'GD Optimal: ', dResults[seed, 7])

            dResults[seed, 11] = learnModel.getCFBound()
            np.savetxt(str(n)+'Results.csv', dResults, delimiter=',')

        meadResults[di,:] = np.mean(dResults, axis=0)
        stdResults[di,:] = np.std(dResults, axis=0)

        np.savetxt('meadResults.csv', meadResults, delimiter=',')
        np.savetxt('stdResults.csv', stdResults, delimiter=',')



    meadResults = np.loadtxt('meadResults.csv', delimiter=',')
    stdResults = np.loadtxt('stdResults.csv', delimiter=',')

    fig, (timeplt) = plt.subplots(1, 1)
    ind = np.arange(numD)  # the x locations for the groups
    width = 0.2  # the width of the bars
    time1 = timeplt.bar(ind - 3 * width / 2, meadResults[:,0], width, yerr=stdResults[:,0],
                    color='Blue', label='MILP-CF')
    time2 = timeplt.bar(ind - width / 2, meadResults[:,1], width, yerr=stdResults[:,1],
                    color='Red', label='MILPBS-CF')
    time3 = timeplt.bar(ind + width / 2, meadResults[:,6], width, yerr=stdResults[:,6],
                    color='Yellow', label='MILP-GD')
    time4 = timeplt.bar(ind + 3 * width / 2, meadResults[:,7], width, yerr=stdResults[:,7],
                    color='Green', label='MILPBS-GD')

    timeplt.set_ylabel('Solution Gap')
    timeplt.set_xticks(ind)
    timeplt.set_xticklabels(dList)
    timeplt.set_xlabel('Number of Data Points')
    timeplt.legend()



    plt.show()
