# -*- coding: utf-8 -*-
import random
import torch
import numpy as np
from torch.autograd import Variable

class NN(torch.nn.Module):
    def __init__(self, K, K2, K3, train):
        super(NN, self).__init__()

        self.train = train
        self.input_linear = list()
        self.input_linear.append(torch.nn.Linear(K, K2, bias=True))
        self.input_linear.append(torch.nn.Linear(K2, K3, bias=True))
        self.input_linear.append(torch.nn.Linear(K3, 1, bias=False))
        self.softmax = torch.nn.Softmax(dim=0)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        out = self.input_linear[0](x)  # N by n by K2
        out = self.tanh(out)  # N by n by K2
        out = self.input_linear[1](out)  # N by n by K3
        out = self.tanh(out)  # N by n by K3
        out = self.input_linear[2](out)  # N by n by 1
        if x.dim() > 1 and not self.train:
            out = self.softmax(out)     # N by n
        return out

class MyLoss(torch.nn.Module):

    def __init__(self):
        super(MyLoss,self).__init__()

    def forward(self,prob,u):
        loss = torch.dot(prob, u)
        return loss


class MyLossReg(torch.nn.Module):

    def __init__(self, mylambda, c, xInit, budget):
        super(MyLossReg,self).__init__()
        self.lagrange = mylambda
        self.c = c
        self.xInit = xInit
        self.budget = budget
        self.relu = torch.nn.ReLU()

    def forward(self,prob,u,x):
        diff = x - self.xInit
        cost = torch.dot(diff.abs().flatten(), self.c.flatten())
        loss = torch.dot(prob, u) + self.lagrange * self.relu(cost - self.budget)
        return loss, cost

class MyLossId(torch.nn.Module):

    def __init__(self):
        super(MyLossId,self).__init__()

    def forward(self,out):
        loss = out
        return loss



class MyLossNeg(torch.nn.Module):

    def __init__(self):
        super(MyLossNeg,self).__init__()

    def forward(self,out):
        loss = -out
        return loss
