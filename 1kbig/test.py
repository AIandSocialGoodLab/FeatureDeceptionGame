from nnGame import Game
from nnSolve import nnSolve
from nnGreedySolve import nnGreedySolve
from nnIpoptSolve import nnIpoptSolve
from learn import LearnNN
import time
import numpy as np
import matplotlib.pyplot as plt
import copy

if __name__ == '__main__':
    seed = 99
    n = 5
    K = 12
    K2 = 24
    K3 = 12
    L = 100
    eps = 0.0001

    numExp = 20
    dList = np.array([10000, 100000, 1000000])
    numD = dList.size

    meadResults = -1 * np.ones((numD,12))
    stdResults = -1 * np.ones((numD,12))
    for di in range(numD):
        numData = int(dList[di])
        dResults = -1 * np.ones((numExp, 12))
        for seed in np.arange(numExp):
            print('numData = ', numData, ', seed = ', seed)
            trueG = Game(seed, n, K, K2, K3)
            learnModel = LearnNN(trueG, numData)
            print('start learning')
            learned_model = learnModel.learn()
            print('finish learning, starting nn solve')


            planG = copy.deepcopy(trueG)
            planG.setModel(learned_model)
            planModel = nnSolve(planG)
            planModel.solve()
            planModel.evaluateSolution(trueG)

            print('nn solved, starting ipopt solve')
            planModelIPOPT = nnIpoptSolve(planG)
            planModelIPOPT.solve()
            planModelIPOPT.evaluateSolution(trueG)
            print('ipopt solved solved')
            planModelGreedy = nnGreedySolve(planG)
            planModelGreedy.solve()
            planModelGreedy.evaluateSolution(trueG)

            dResults[seed, 0] = planModel.trueoptVal
            dResults[seed, 1] = planModelIPOPT.trueoptVal
            dResults[seed, 2] = planModelGreedy.trueoptVal

            dResults[seed, 3] = np.abs(planModel.trueoptVal - planModel.optVal) / planModel.trueoptVal
            dResults[seed, 4] = np.abs(planModelIPOPT.trueoptVal - planModelIPOPT.optVal) / planModelIPOPT.trueoptVal
            dResults[seed, 5] = np.abs(planModelGreedy.trueoptVal - planModelGreedy.optVal) / planModelGreedy.trueoptVal



            truePlanModel = nnSolve(trueG)
            truePlanModel.solve()
            dResults[seed, 0] = (dResults[seed, 0] - truePlanModel.optVal) / truePlanModel.optVal
            dResults[seed, 1] = (dResults[seed, 1] - truePlanModel.optVal) / truePlanModel.optVal
            dResults[seed, 2] = (dResults[seed, 2] - truePlanModel.optVal) / truePlanModel.optVal

            np.savetxt(str(numData)+'Results.csv', dResults, delimiter=',')

        meadResults[di,:] = np.mean(dResults, axis=0)
        stdResults[di,:] = np.std(dResults, axis=0)

        np.savetxt('meadResults.csv', meadResults, delimiter=',')
        np.savetxt('stdResults.csv', stdResults, delimiter=',')



    meadResults = np.loadtxt('meadResults.csv', delimiter=',')
    stdResults = np.loadtxt('stdResults.csv', delimiter=',')

    fig, (timeplt) = plt.subplots(1, 1)
    ind = np.arange(numD)  # the x locations for the groups
    width = 0.2  # the width of the bars
    time1 = timeplt.bar(ind - width, meadResults[:numD,0], width, yerr=stdResults[:numD,0],
                    color='Blue', label='GD')
    time2 = timeplt.bar(ind, meadResults[:numD,1], width, yerr=stdResults[:numD,1],
                    color='Red', label='IPOPT')
    time3 = timeplt.bar(ind + width, meadResults[:numD,2], width, yerr=stdResults[:numD,2],
                    color='Yellow', label='Greedy')


    timeplt.set_ylabel('Solution Gap')
    timeplt.set_xticks(ind)
    timeplt.set_xticklabels(dList)
    timeplt.set_xlabel('Number of Data Points')
    timeplt.legend()

    plt.show()
