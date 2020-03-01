from nn import NN
import numpy as np
import torch
import copy

class LearnNN:
    def __init__(self, game, numData):
        self.n = game.n
        self.K = game.K
        self.K2 = game.K2
        self.K3 = game.K3
        self.seed = game.seed
        self.numData = numData

        self.Ntest = 500000
        self.lr = 1e-1
        self.nepoch = 20
        self.nstep = 10
        self.lb = -1
        self.ub = 1

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.target = copy.deepcopy(game.f)
        self.target.train = True
        self.learn_model = NN(game.K, game.K2, game.K3, True)
        self.history = np.zeros(4)

        self.xTest = torch.randn(self.Ntest, self.n, self.K)
        testScore = self.target.forward(self.xTest)
        testProb = torch.nn.functional.softmax(testScore, dim=1)
        self.yTest = torch.squeeze(torch.multinomial(testProb, 1))



    def learn(self):
        self.Ntrain = self.numData
        self.batch_size = int(self.Ntrain / self.nepoch)
        self.learn_model = NN(self.K, self.K2, self.K3, True)
        self.learnGD()
        np.savetxt("history"+str(self.n)+"_"+str(self.K)+"_"+str(self.K2)+"_"+str(self.K3)+"_"+str(self.seed)+".csv", self.history, delimiter=',')
        return self.learn_model




    def learnGD(self):
        xTrain = torch.randn(self.Ntrain, self.n, self.K)
        trueScore = self.target.forward(xTrain)
        trueProb = torch.nn.functional.softmax(trueScore, dim=1)
        yTrain = torch.squeeze(torch.multinomial(trueProb, 1))

        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        criterionTest = torch.nn.L1Loss(reduction='mean')
        optimizer = torch.optim.RMSprop(self.learn_model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

        for epoch in range(self.nepoch):
            scheduler.step()
            for t in range(self.nstep):
                rand_ix = np.random.randint(0, self.Ntrain, (self.batch_size,))
                batch_x = xTrain[rand_ix]
                batch_y = yTrain[rand_ix]
                y_pred = self.learn_model(batch_x)
                loss = criterion(y_pred, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            y_pred_test = self.learn_model(self.xTest)
            loss_test = criterion(y_pred_test, self.yTest)
            y_pred_validation = self.target(self.xTest)
            loss_validation = criterion(y_pred_validation, self.yTest)
            optimizer.zero_grad()

            y_pred_test_prob = torch.nn.functional.softmax(y_pred_test, dim=1)
            y_pred_validation_prob = torch.nn.functional.softmax(y_pred_validation, dim=1)
            TV_loss = criterionTest(y_pred_test_prob, y_pred_validation_prob)*self.n/2

        L1distance = 0
        for i in range(len(list(self.learn_model.parameters()))):
            w = np.array(list(self.learn_model.parameters())[i].data)
            target_w = np.array(list(self.target.parameters())[i].data)
            L1distance += np.sum(np.abs(w - target_w))
        self.history = np.array([self.Ntrain, loss_test.item()/self.Ntest - loss_validation.item()/self.Ntest, TV_loss.item(), L1distance])
        return self.learn_model
