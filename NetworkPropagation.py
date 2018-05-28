import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class NetworkPropagation():
    class PropagationNetwork(nn.Module):
        def __init__(self, sources, targets, nIter, nHiddens, nFeatures, pDropout, adaptiveEdgeWeights):
            super(NetworkPropagation.PropagationNetwork, self).__init__()
            self.dropout = nn.Dropout(pDropout)
            self.nIter = nIter
            self.dense0 = nn.Linear(1, nHiddens)
            self.dense1 = nn.Linear(1, nHiddens)
            self.dense2 = nn.Linear(nHiddens, nFeatures)
            self.dense3 = nn.Linear(nFeatures, nHiddens)
            self.dense4 = nn.Linear(nHiddens, 1)
            
        def forward(self, x, mask, sparse):
            x = self.dropout(x)
            for i in range(self.nIter):
                x = self.dense4(F.sigmoid(self.dense0(x) + self.dense3(torch.matmul(sparse, self.dense2(F.sigmoid(self.dense1(x)))))))
            return x * mask
    
    def __init__(self, values, sources, targets, lossFn, optimizer, lr = 0.001, nIter = 3, nHiddens = 128, nFeatures = 32, gpu = True, pHoldout = 0.2, pDropout = 0.5, adaptiveEdgeWeights = False):
        n = len(values)
        self.nn = self.PropagationNetwork(sources, targets, nIter = nIter, nHiddens = nHiddens, nFeatures = nFeatures, pDropout = pDropout, adaptiveEdgeWeights = adaptiveEdgeWeights)
        y = torch.t(torch.FloatTensor([values]))
        nHoldout = int(len(values) * pHoldout)
        mask = np.concatenate((np.ones((n - nHoldout, 1)), np.zeros((nHoldout, 1))))
        np.random.shuffle(mask)
        trainMask = torch.FloatTensor(mask.tolist())
        holdoutMask = torch.FloatTensor((np.ones((n, 1)) - mask).tolist())
        self.x = Variable(y * trainMask, requires_grad = False)
        self.y = Variable(y * holdoutMask, requires_grad = False)
        self.trainMask = Variable(trainMask, requires_grad = False)
        self.holdoutMask = Variable(holdoutMask, requires_grad = False)
        self.lossFn = lossFn
        self.optimizer = optimizer(self.nn.parameters(), lr = lr)
        self.noMask = Variable(torch.FloatTensor((np.ones((n, 1))).tolist()), requires_grad = False)
        count = [0] * n
        for v in sources:
            count[v] += 1
        weights = [1.0 / count[v] for v in sources]
        self.sparse = Variable(torch.sparse.FloatTensor(torch.LongTensor([targets, sources]), torch.FloatTensor(weights)), requires_grad = adaptiveEdgeWeights) #TODO: divide by cardinality
        
        if gpu:
            self.x = self.x.cuda()
            self.y = self.y.cuda()
            self.trainMask = self.trainMask.cuda()
            self.holdoutMask = self.holdoutMask.cuda()
            self.noMask = self.noMask.cuda()
            self.nn.cuda()
            self.sparse = self.sparse.cuda()
    
    def saveNetwork(self, path):
        torch.save(self.nn, path)
        
    def loadNetwork(self, path):
        self.nn = torch.load(path)
        
    def train(self, nIter, earlyStopThreshold = 10):
        self.nn.train()
        minIter = 0
        minLoss = 100
        for i in range(nIter):
            self.optimizer.zero_grad()
            loss = self.lossFn(self.nn(self.x, self.trainMask, self.sparse), self.x)
            trainLoss = loss.item()
            loss.backward()
            self.optimizer.step()
            loss = self.lossFn(self.nn(self.x, self.holdoutMask, self.sparse), self.y)
            holdoutLoss = loss.item()
            print("Epoch =", i + 1)
            print("Training Loss =", trainLoss)
            print("Validation Loss =", holdoutLoss)
            print()
            if holdoutLoss < minLoss:
                minLoss = holdoutLoss
                minIter = i
            if i >= minIter + earlyStopThreshold:
                break
    
    def test(self):
        self.nn.eval()
        return self.nn(self.x, self.noMask, self.sparse)
