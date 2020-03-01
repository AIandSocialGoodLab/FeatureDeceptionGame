from learn import LearnNN
from game import Game
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    seed = 99
    n = 5
    K = 12
    K2 = 30
    K3 = 50
    nkset = [(5, 12), (10, 12), (5, 24)]
    numExp = 20
    numData = np.linspace(120, 2280, 10, dtype=int)
    diffBound = -1 * np.ones((3, numExp))
    resultsCF = np.zeros((3, numExp, numData.size, 4))
    resultsGD = np.zeros((3, numExp, numData.size, 4))

    for nki in range(len(nkset)):
        n = nkset[nki][0]
        K = nkset[nki][1]

        for seed in range(numExp):
            print('n: ', n, 'K: ', K, 'seed: ', seed)
            game = Game(seed, n, K)
            model = LearnNN(seed, n, K, K2, K3, game.f, numData)
            model.learn()

            resultsCF[nki,seed,:,:] = model.historyCF
            resultsGD[nki,seed,:,:] = model.historyGD
            diffBound[nki,seed] = model.bound


        np.save("resultsCF"+str(n)+"_"+str(K)+".npy", resultsCF[nki,:,:,:])
        np.save("resultsGD"+str(n)+"_"+str(K)+".npy", resultsGD[nki,:,:,:])
        np.save("diffBound"+str(n)+"_"+str(K)+".npy", diffBound[nki,:])

    np.save("resultsCFall.npy", resultsCF)
    np.save("resultsGDall.npy", resultsGD)
    np.save("diffBoundall.npy", diffBound)




    colors = ['#FCB711', '#F37021', '#CC004C', '#6460AA', '#0089D0', '#0DB14B']

    resultsCF = np.load("resultsCFall.npy")
    resultsGD = np.load("resultsGDall.npy")
    diffBound = np.load("diffBoundall.npy")


    lossGDmean = np.mean(resultsGD[:,:,:,2], axis=1)
    lossCFmean = np.mean(resultsCF[:,:,:,2], axis=1)

    diffGDmean = np.mean(resultsGD[:,:,:,3], axis=1)
    diffCFmean = np.mean(resultsCF[:,:,:,3], axis=1)

    lossGDstd = np.std(resultsGD[:,:,:,2], axis=1)
    lossCFstd = np.std(resultsCF[:,:,:,2], axis=1)

    diffGDstd = np.std(resultsGD[:,:,:,3], axis=1)
    diffCFstd = np.std(resultsCF[:,:,:,3], axis=1)


    fig, (lossplt, diffplt) = plt.subplots(2, 1)

    lossGDline1 = lossplt.errorbar(numData, lossGDmean[0,:], yerr=lossGDstd[0,:], color=colors[0],
     fmt='o-', label='RMSProp, n = '+str(nkset[0][0])+', K = '+str(nkset[0][1]), capsize=2)
    lossCFline1 = lossplt.errorbar(numData, lossCFmean[0,:], yerr=lossCFstd[0,:], color=colors[1],
     fmt='D--', label='CF, n = '+str(nkset[0][0])+', K = '+str(nkset[0][1]), capsize=2)

    diffGDline1 = diffplt.errorbar(numData, diffGDmean[0,:], yerr=diffGDstd[0,:], color=colors[0],
     fmt='o-', label='RMSProp, n = '+str(nkset[0][0])+', K = '+str(nkset[0][1]), capsize=2)
    diffCFline1 = diffplt.errorbar(numData, diffCFmean[0,:], yerr=diffCFstd[0,:], color=colors[1],
     fmt='D--', label='CF, n = '+str(nkset[0][0])+', K = '+str(nkset[0][1]), capsize=2)

    lossGDline2 = lossplt.errorbar(numData, lossGDmean[1,:], yerr=lossGDstd[1,:], color=colors[2],
     fmt='o-', label='RMSProp, n = '+str(nkset[1][0])+', K = '+str(nkset[1][1]), capsize=2)
    lossCFline2 = lossplt.errorbar(numData, lossCFmean[1,:], yerr=lossCFstd[1,:], color=colors[3],
     fmt='D--', label='CF, n = '+str(nkset[1][0])+', K = '+str(nkset[1][1]), capsize=2)

    diffGDline2 = diffplt.errorbar(numData, diffGDmean[1,:], yerr=diffGDstd[1,:], color=colors[2],
     fmt='o-', label='RMSProp, n = '+str(nkset[1][0])+', K = '+str(nkset[1][1]), capsize=2)
    diffCFline2 = diffplt.errorbar(numData, diffCFmean[1,:], yerr=diffCFstd[1,:], color=colors[3],
     fmt='D--', label='CF, n = '+str(nkset[1][0])+', K = '+str(nkset[1][1]), capsize=2)

    lossGDline3 = lossplt.errorbar(numData, lossGDmean[2,:], yerr=lossGDstd[0,:], color=colors[4],
     fmt='o-', label='RMSProp, n = '+str(nkset[2][0])+', K = '+str(nkset[2][1]), capsize=2)
    lossCFline3 = lossplt.errorbar(numData, lossCFmean[2,:], yerr=lossCFstd[0,:], color=colors[5],
     fmt='D--', label='CF, n = '+str(nkset[2][0])+', K = '+str(nkset[2][1]), capsize=2)

    diffGDline3 = diffplt.errorbar(numData, diffGDmean[2,:], yerr=diffGDstd[2,:], color=colors[4],
     fmt='o-', label='RMSProp, n = '+str(nkset[2][0])+', K = '+str(nkset[2][1]), capsize=2)
    diffCFline3 = diffplt.errorbar(numData, diffCFmean[2,:], yerr=diffCFstd[2,:], color=colors[5],
     fmt='D--', label='CF, n = '+str(nkset[2][0])+', K = '+str(nkset[2][1]), capsize=2)

    lossplt.set_xlabel('Number of Data Points')
    lossplt.set_ylabel('Attack Probability Error')

    diffplt.set_xlabel('Number of Data Points')
    diffplt.set_ylabel('L1-norm Parameter Difference')

    losslegend = lossplt.legend(loc='upper right')
    difflegend = diffplt.legend(loc='upper right')


    plt.show()
