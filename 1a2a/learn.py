from nn import NN
import numpy as np
import torch

class LearnNN:
    def __init__(self, seed, n, K, K2, K3, target, numData):
        self.n = n
        self.K = K
        self.K2 = K2
        self.K3 = K3
        self.seed = seed
        self.numData = numData

        self.Nstrat = self.K
        self.Ntest = 500000
        self.lr = 1e-1
        self.nepoch = 20
        self.nstep = 10
        self.lb = -1
        self.ub = 1


        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.target = target

        self.CF_model = NN(K, K2, K3, True)
        self.GD_model = NN(K, K2, K3, True)
        self.historyCF = np.zeros((self.numData.size, 4))
        self.historyGD = np.zeros((self.numData.size, 4))

        self.xTest = torch.randn(self.Ntest, self.n, self.K)
        testScore = self.target.forward(self.xTest)
        testProb = torch.nn.functional.softmax(testScore, dim=1)
        self.yTest = torch.squeeze(torch.multinomial(testProb, 1))

    def learn(self):
        self.bound = self.getCFBound()
        for datasizei in range(self.numData.size):
            self.Ntrain = self.numData[datasizei]
            self.batch_size = int(self.Ntrain / self.nepoch)
            self.GD_model = NN(self.K, self.K2, self.K3, True)
            self.learnGD(datasizei)
            np.savetxt("historyGD"+str(self.n)+"_"+str(self.K)+"_"+str(self.seed)+".csv", self.historyGD, delimiter=',')
        for datasizei in range(self.numData.size):
            self.Ntrain = self.numData[datasizei]
            self.CF_model = NN(self.K, self.K2, self.K3, True)
            self.learnCF(datasizei)
            np.savetxt("historyCF"+str(self.n)+"_"+str(self.K)+"_"+str(self.seed)+".csv", self.historyCF, delimiter=',')

    def learnCF(self, sizei):
        np.random.seed(self.seed)
        xTrain = np.random.rand(self.K, self.n, self.K)
        f1 = 1
        f0 = 0
        for i in range(self.K):
            xTrain[i,f1,:] = xTrain[i,f0,:]
            xTrain[i,f1,i] = 0
            xTrain[i,f0,i] = 1
        xTrain = np.repeat(xTrain, int(self.Ntrain / self.K), 0)
        xTrain = torch.tensor(xTrain, dtype=torch.float)
        trueScore = self.target.forward(xTrain)
        trueProb = torch.nn.functional.softmax(trueScore, dim=1)
        yTrain = torch.squeeze(torch.multinomial(trueProb, 1))
        yTrainByStrat = yTrain.reshape(self.K, -1)
        bst = -1 * np.ones(self.K)
        Ast = -1 * np.ones((self.K, self.K))
        counts = list()
        rho = 1
        for i in range(self.K):
            unique, count = np.unique(yTrainByStrat[i,:], return_counts=True)
            countdict = dict(zip(unique, count))
            if f0 not in countdict:
                f0count = 0.1
            else:
                f0count = countdict[f0]
            if f1 not in countdict:
                f1count = 0.1
            else:
                f1count = countdict[f1]
            bst[i] = np.log(f0count/f1count)
            for j in range(self.K):
                Ast[i,j] = xTrain[int(self.Ntrain / self.K) * i, f0, j] - xTrain[int(self.Ntrain / self.K) * i, f1, j]
        alpha = np.linalg.norm(np.linalg.inv(Ast), ord=np.inf)
        w = np.linalg.solve(Ast, bst)

        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        criterionTest = torch.nn.L1Loss(reduction='mean')
        with torch.no_grad():
            self.CF_model.input_linear.weight = torch.nn.Parameter(torch.unsqueeze(torch.tensor(w, dtype=torch.float), dim=0))
            y_pred_test = self.CF_model(self.xTest)
            loss_test = criterion(y_pred_test, self.yTest)
            y_pred_validation = self.target(self.xTest)
            loss_validation = criterion(y_pred_validation, self.yTest)

            y_pred_test_prob = torch.nn.functional.softmax(y_pred_test, dim=1)
            y_pred_validation_prob = torch.nn.functional.softmax(y_pred_validation, dim=1)
            TV_loss = criterionTest(y_pred_test_prob, y_pred_validation_prob)*self.n/2
        target_w = np.array(self.target.input_linear.weight.data)
        self.historyCF[sizei,:] = np.array([self.Ntrain, loss_test.item()/self.Ntest - loss_validation.item()/self.Ntest, TV_loss.item(), np.mean(np.abs(w - target_w))])

    def learnGD(self, sizei):
        np.random.seed(self.seed)
        xTrain = np.random.rand(self.K, self.n, self.K)
        for i in range(self.K):
            xTrain[i,1,:] = xTrain[i,0,:]
            xTrain[i,1,i] = 0
            xTrain[i,0,i] = 1
        xTrain = torch.tensor(xTrain, dtype=torch.float)
        xTrain = np.repeat(xTrain, int(self.Ntrain / self.K), 0)
        trueScore = self.target.forward(xTrain)
        trueProb = torch.nn.functional.softmax(trueScore, dim=1)
        yTrain = torch.squeeze(torch.multinomial(trueProb, 1))


        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        criterionTest = torch.nn.L1Loss(reduction='mean')
        optimizer = torch.optim.RMSprop(self.GD_model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
        for epoch in range(self.nepoch):
            scheduler.step()
            for t in range(self.nstep):
                rand_ix = np.random.randint(0, self.Ntrain, (self.batch_size,))
                batch_x = xTrain[rand_ix]
                batch_y = yTrain[rand_ix]
                y_pred = self.GD_model(batch_x)

                loss = criterion(y_pred, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                y_pred_test = self.GD_model(self.xTest)
                loss_test = criterion(y_pred_test, self.yTest)
                y_pred_validation = self.target(self.xTest)
                loss_validation = criterion(y_pred_validation, self.yTest)
                optimizer.zero_grad()

                y_pred_test_prob = torch.nn.functional.softmax(y_pred_test, dim=1)
                y_pred_validation_prob = torch.nn.functional.softmax(y_pred_validation, dim=1)
                TV_loss = criterionTest(y_pred_test_prob, y_pred_validation_prob)*self.n/2
        target_w = np.array(self.target.input_linear.weight.data)
        w = np.array(self.GD_model.input_linear.weight.data)
        self.historyGD[sizei,:] = np.array([self.Ntrain, loss_test.item()/self.Ntest - loss_validation.item()/self.Ntest, TV_loss.item(), np.mean(np.abs(w - target_w))])
        return self.GD_model

    def getCFBound(self):
        target_w = np.array(self.target.input_linear.weight.data)
        maxScore = np.sum(np.clip(target_w, a_min=0, a_max=None))
        minScore = np.sum(np.clip(target_w, a_max=0, a_min=None))
        maxScore = np.exp(maxScore)
        minScore = np.exp(minScore)
        self.rho = minScore / (minScore + (self.n-1) * maxScore)
        eps = 10
        delta = 0.01
        self.bound = np.power(self.K, 4) / (self.rho * eps * eps) * np.log(self.n * self.K / delta)
        np.savetxt("boundCF"+str(self.n)+"_"+str(self.K)+"_"+str(self.seed)+".csv", np.array([self.bound]), delimiter=',')
        return self.bound
