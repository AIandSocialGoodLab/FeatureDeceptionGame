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
    K = 12
    K2 = 24
    K3 = 12
    L = 100
    eps = 0.0001

    numExp = 20
    nList = np.concatenate([np.linspace(2,10,5,dtype=int), [20, 50]])
    numN = nList.size

    meanResults = -1 * np.ones((numN,12))
    stdResults = -1 * np.ones((numN,12))
    for ni in range(numN):
        n = int(nList[ni])
        nResults = -1 * np.ones((numExp, 12))
        for seed in np.arange(numExp):
            print('n = ', n, ', seed = ', seed)
            trueG = Game(seed, n, K, K2, K3)

            learnModel = LearnNN(trueG)
            print('start learning')
            learned_model = learnModel.learn()
            print('finish learning, starting nn solve')

            # Test closed form learning
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

            nResults[seed, 0] = planModel.trueoptVal
            nResults[seed, 1] = planModelIPOPT.trueoptVal
            nResults[seed, 2] = planModelGreedy.trueoptVal

            nResults[seed, 3] = -(planModel.trueoptVal - planModel.trueoriVal) / planModel.trueoriVal
            nResults[seed, 4] = -(planModelIPOPT.trueoptVal - planModelIPOPT.trueoriVal) / planModelIPOPT.trueoriVal
            nResults[seed, 5] = -(planModelGreedy.trueoptVal - planModelGreedy.trueoriVal) / planModelGreedy.trueoriVal



            truePlanModel = nnSolve(trueG)
            truePlanModel.solve()
            truePlanModel.evaluateSolution(trueG)

            nResults[seed, 0] = (nResults[seed, 0] - truePlanModel.optVal) / truePlanModel.optVal
            nResults[seed, 1] = (nResults[seed, 1] - truePlanModel.optVal) / truePlanModel.optVal
            nResults[seed, 2] = (nResults[seed, 2] - truePlanModel.optVal) / truePlanModel.optVal

        meanResults[ni,:] = np.mean(nResults, axis=0)
        stdResults[ni,:] = np.std(nResults, axis=0)

        np.savetxt('meanResults.csv', meanResults, delimiter=',')
        np.savetxt('stdResults.csv', stdResults, delimiter=',')



    meanResults = np.loadtxt('meanResults.csv', delimiter=',')
    stdResults = np.loadtxt('stdResults.csv', delimiter=',')

    fig, (timeplt) = plt.subplots(1, 1)
    ind = np.arange(numN)  # the x locations for the groups
    width = 0.2  # the width of the bars
    time1 = timeplt.bar(ind - width, meanResults[:numN,0], width, yerr=stdResults[:numN,0],
                    color='Blue', label='GD')
    time2 = timeplt.bar(ind, meanResults[:numN,1], width, yerr=stdResults[:numN,1],
                    color='Red', label='IPOPT')
    time3 = timeplt.bar(ind + width, meanResults[:numN,2], width, yerr=stdResults[:numN,2],
                    color='Yellow', label='Greedy')


    timeplt.set_ylabel('Solution Gap')
    timeplt.set_xticks(ind)
    timeplt.set_xticklabels(nList)
    timeplt.set_xlabel('Number of targets')
    timeplt.legend()
    plt.show()
