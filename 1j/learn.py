from nn import NN
import numpy as np
import torch

class LearnNN:
    def __init__(self, game):
        self.n = game.n
        self.K = game.K
        self.seed = game.seed

        self.Nstrat = self.K
        self.Ntrain = 1000000
        self.Ntest = 500000
        self.lr = 1e-1
        self.nepoch = 30
        self.nstep = self.K
        self.batch_size = 5000
        self.lb = -1
        self.ub = 1

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.target = game.f

        self.CF_model = NN(self.K, 10, 20, True)
        self.GD_model = NN(self.K, 10, 20, True)
        self.CF_history = np.zeros((self.nepoch, 4))
        self.GD_history = np.zeros((self.nepoch, 4))

    def learnCF(self):
        xTest = torch.randn(self.Ntest, self.n, self.K)
        testScore = self.target.forward(xTest)
        testProb = torch.nn.functional.softmax(testScore, dim=1)
        yTest = torch.squeeze(torch.multinomial(testProb, 1))
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')

        for epoch in range(self.nepoch):
            np.random.seed(self.seed)
            xTrain = np.random.rand(self.K, self.n, self.K)
            for i in range(self.K):
                xTrain[i,1,:] = xTrain[i,0,:]
                xTrain[i,1,i] = 0
                xTrain[i,0,i] = 1
            xTrain = np.repeat(xTrain, self.batch_size * (epoch + 1), 0)
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
                rho = min(rho, min(count)/(self.batch_size * (epoch + 1)))
                bst[i] = np.log(count[0]/count[1])
                for j in range(self.K):
                    Ast[i,j] = xTrain[self.batch_size * (epoch+1) * i, 0, j] - xTrain[self.batch_size * (epoch+1) * i, 1, j]
            alpha = np.linalg.norm(np.linalg.inv(Ast), ord=np.inf)
            eps = 1
            delta = 0.01
            self.bound = np.power(alpha, 4) * np.power(self.K, 4) / (rho * eps * eps) * np.log(self.n * self.K / delta)
            w = np.linalg.solve(Ast, bst)


            with torch.no_grad():
                self.CF_model.input_linear.weight = torch.nn.Parameter(torch.unsqueeze(torch.tensor(w, dtype=torch.float), dim=0))
                y_pred_test = self.CF_model(xTest)
                loss_test = criterion(y_pred_test, yTest)
                y_pred_validation = self.target(xTest)
                loss_validation = criterion(y_pred_validation, yTest)
                target_w = np.array(self.target.input_linear.weight.data)
                self.CF_history[epoch,:] = np.array([epoch, loss_test.item()/self.Ntest, loss_validation.item()/self.Ntest, np.sum(np.abs(w - target_w))])
            return self.CF_model

    def learnGD(self):
        xTest = torch.randn(self.Ntest, self.n, self.K)
        testScore = self.target.forward(xTest)
        testProb = torch.nn.functional.softmax(testScore, dim=1)
        yTest = torch.squeeze(torch.multinomial(testProb, 1))


        xTrain = torch.randn(self.Ntrain, self.n, self.K)
        trueScore = self.target.forward(xTrain)
        trueProb = torch.nn.functional.softmax(trueScore, dim=1)
        yTrain = torch.squeeze(torch.multinomial(trueProb, 1))

        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        optimizer = torch.optim.RMSprop(self.GD_model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
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
                y_pred_test = self.GD_model(xTest)
                loss_test = criterion(y_pred_test, yTest)
                y_pred_validation = self.target(xTest)
                loss_validation = criterion(y_pred_validation, yTest)
                optimizer.zero_grad()
                diff = torch.nn.functional.softmax(y_pred_test, dim=1) - torch.nn.functional.softmax(y_pred_validation, dim=1)
                predProb = torch.nn.functional.softmax(y_pred_test, dim=1).detach().numpy()
                trueProb = torch.nn.functional.softmax(y_pred_validation, dim=1).detach().numpy()

                target_w = np.array(self.target.input_linear.weight.data)
                w = np.array(self.GD_model.input_linear.weight.data)
                self.GD_history[epoch,:] = np.array([epoch, loss_test.item()/self.Ntest, loss_validation.item()/self.Ntest, np.sum(np.abs(w - target_w))])
        return self.GD_model

    def getCFBound(self):
        return self.bound
