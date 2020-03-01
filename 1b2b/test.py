from learn import LearnNN
from nnGame import Game
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    seed = 99
    n = 10
    K = 12
    K2 = 24
    K3 = 12

    nkset = [(5, 12), (10, 12), (5, 24)]
    numExp = 20
    numData = np.linspace(120, 2280, 10, dtype=int)
    results = np.zeros((3, numExp, numData.size, 4))

    for nki in range(len(nkset)):
        n = nkset[nki][0]
        K = nkset[nki][1]

        for seed in range(numExp):
            print('n: ', n, 'K: ', K, 'seed: ', seed)
            game = Game(seed, n, K, K2, K3)
            model = LearnNN(seed, n, K, K2, K3, game.f, numData)
            model.learn()

            results[nki,seed,:,:] = model.history

            np.save("results"+str(n)+"_"+str(K)+"_"+str(K2)+"_"+str(K3)+".npy", results[nki,:,:,:])

    np.save("resultsall.npy", results)






    results = np.load("resultsall.npy")


    lossmean = np.mean(results[:,:,:,2], axis=1)
    diffmean = np.mean(results[:,:,:,3], axis=1)
    lossstd = np.std(results[:,:,:,2], axis=1)
    diffstd = np.std(results[:,:,:,3], axis=1)


    fig, (lossplt, diffplt) = plt.subplots(2, 1)

    lossline1 = lossplt.errorbar(numData, lossmean[0,:], yerr=lossstd[0,:],
     fmt='o-', label='Gradient Descent, n = '+str(nkset[0][0])+', K = '+str(nkset[0][1]), capsize=2)

    diffline1 = diffplt.errorbar(numData, diffmean[0,:], yerr=diffstd[0,:],
     fmt='o-', label='Gradient Descent, n = '+str(nkset[0][0])+', K = '+str(nkset[0][1]), capsize=2)

    lossline2 = lossplt.errorbar(numData, lossmean[1,:], yerr=lossstd[1,:],
     fmt='o-', label='Gradient Descent, n = '+str(nkset[1][0])+', K = '+str(nkset[1][1]), capsize=2)

    diffline2 = diffplt.errorbar(numData, diffmean[1,:], yerr=diffstd[1,:],
     fmt='o-', label='Gradient Descent, n = '+str(nkset[1][0])+', K = '+str(nkset[1][1]), capsize=2)

    lossline3 = lossplt.errorbar(numData, lossmean[2,:], yerr=lossstd[0,:],
     fmt='o-', label='Gradient Descent, n = '+str(nkset[2][0])+', K = '+str(nkset[2][1]), capsize=2)

    diffline3 = diffplt.errorbar(numData, diffmean[2,:], yerr=diffstd[2,:],
     fmt='o-', label='Gradient Descent, n = '+str(nkset[2][0])+', K = '+str(nkset[2][1]), capsize=2)

    lossplt.set_xlabel('Number of Data Points')
    lossplt.set_ylabel('Attack Probability Error')

    diffplt.set_xlabel('Number of Data Points')
    diffplt.set_ylabel('L1-norm Parameter Difference')

    losslegend = lossplt.legend(loc='upper right')
    difflegend = diffplt.legend(loc='upper right')


    plt.show()
