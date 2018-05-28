import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class NetworkPropagation():
    class PropagationNetwork(nn.Module):
        def __init__(self, nNodes, sources, targets, nIter, nHiddens, nFeatures, gpu, pDropout, adaptiveEdgeWeights):
            super(NetworkPropagation.PropagationNetwork, self).__init__()
            count = [0] * nNodes
            for v in sources:
                count[v] += 1
            weights = [1.0 / count[v] for v in sources]
            self.dropout = nn.Dropout(pDropout)
            self.nIter = nIter
            self.dense0 = nn.Linear(1, nHiddens)
            self.dense1 = nn.Linear(1, nHiddens)
            self.dense2 = nn.Linear(nHiddens, nFeatures)
            self.sparse = Variable(torch.sparse.FloatTensor(torch.LongTensor([targets, sources]), torch.FloatTensor(weights)), requires_grad = adaptiveEdgeWeights) #TODO: divide by cardinality
            self.dense3 = nn.Linear(nFeatures, nHiddens)
            self.dense4 = nn.Linear(nHiddens, 1)
            if gpu:
                self.sparse = self.sparse.cuda()
            
        def forward(self, x, mask):
            x = self.dropout(x)
            for i in range(self.nIter):
                x = self.dense4(F.sigmoid(self.dense0(x) + self.dense3(torch.matmul(self.sparse, self.dense2(F.sigmoid(self.dense1(x)))))))
            return x * mask
    
    def __init__(self, values, sources, targets, loss_fn, optimizer, lr = 0.0005, nIter = 3, nHiddens = 128, nFeatures = 32, gpu = True, pHoldout = 0.2, pDropout = 0.5, adaptiveEdgeWeights = False):
        n = len(values)
        self.nn = self.PropagationNetwork(n, sources, targets, nIter = nIter, nHiddens = nHiddens, nFeatures = nFeatures, gpu = gpu, pDropout = pDropout, adaptiveEdgeWeights = adaptiveEdgeWeights)
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
        self.loss_fn = loss_fn
        self.optimizer = optimizer(self.nn.parameters(), lr = lr)
        if gpu:
            self.x = self.x.cuda()
            self.y = self.y.cuda()
            self.trainMask = self.trainMask.cuda()
            self.holdoutMask = self.holdoutMask.cuda()
            self.nn.cuda()
        
    def train(self, nIter):
        for i in range(nIter):
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.nn(self.x, self.trainMask), self.x)
            train_loss = loss.item()
            loss.backward()
            self.optimizer.step()
            loss = self.loss_fn(self.nn(self.x, self.holdoutMask), self.y)
            holdout_loss = loss.item()
            print("Epoch =", i + 1)
            print("Training Loss =", train_loss)
            print("Validation Loss =", holdout_loss)
            print()

netprop = NetworkPropagation(np.random.rand(10000), np.random.randint(10000, size=1000000).tolist(), np.random.randint(10000, size=1000000).tolist(), loss_fn = F.mse_loss, optimizer = torch.optim.Adam)
netprop.train(100)