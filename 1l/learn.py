from nn import NN
import numpy as np
import torch
import copy

class LearnNN:
    def __init__(self, game):
        self.n = game.n
        self.K = game.K
        self.K2 = game.K2
        self.K3 = game.K3
        self.seed = game.seed

        self.Ntrain = 1000000
        self.Ntest = 500000
        self.lr = 1e-1
        self.nepoch = 30
        self.nstep = 30
        self.batch_size = 5000
        self.lb = -1
        self.ub = 1

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.target = copy.deepcopy(game.f)
        self.target.train = True
        self.learn_model = NN(game.K, game.K2, game.K3, True)
        self.history = np.zeros(4)


    def learn(self):

        xTest = torch.randn(self.Ntest, self.n, self.K)
        testScore = self.target.forward(xTest)
        testProb = torch.nn.functional.softmax(testScore, dim=1)
        yTest = torch.squeeze(torch.multinomial(testProb, 1))

        xTrain = torch.randn(self.Ntrain, self.n, self.K)
        trueScore = self.target.forward(xTrain)
        trueProb = torch.nn.functional.softmax(trueScore, dim=1)
        yTrain = torch.squeeze(torch.multinomial(trueProb, 1))

        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        optimizer = torch.optim.RMSprop(self.learn_model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        criterionTest = torch.nn.L1Loss(reduction='mean')

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
            y_pred_test = self.learn_model(xTest)
            loss_test = criterion(y_pred_test, yTest)
            y_pred_validation = self.target(xTest)
            loss_validation = criterion(y_pred_validation, yTest)
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
        print(',test loss: ', loss_test.item()/self.Ntest, ', validation loss: ', loss_validation.item()/self.Ntest, 'TV loss:', TV_loss.item())
        return self.learn_model