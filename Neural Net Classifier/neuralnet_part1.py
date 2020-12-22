import numpy as np
import torch

class NeuralNet(torch.nn.Module):
    def __init__(self, lrate,loss_fn,in_size, out_size):
        super(NeuralNet, self).__init__()
        self.fc1 = torch.nn.Linear(in_size, 64)
        self.fc2 = torch.nn.Linear(64, out_size)
        self.lrate = lrate
        self.loss_fn = loss_fn
        self.optim = torch.optim.SGD(self.parameters(), lr=self.lrate, momentum=0.97)

    def forward(self, x):
        return torch.relu(self.fc2(torch.relu(self.fc1(x))))
    
    def step(self,x,y):
        self.optim.zero_grad()
        out = self.forward(x)
        L = self.loss_fn(out, y)
        L.backward()
        self.optim.step()
        return L.item()
    
def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    lrate = 1e-3
    loss_fn = torch.nn.CrossEntropyLoss()
    net = NeuralNet(lrate, loss_fn, train_set.shape[1], 2)
    losses = []
    yhats = []
    for i,data in enumerate(train_set):
        avg = data.mean(dim=0)
        stdev = data.std(dim=0)
        train_set[i] = (data - avg) / stdev
    for i, data in enumerate(dev_set):
        avg = data.mean(dim=0)
        stdev = data.std(dim=0)
        dev_set[i] = (data - avg) / stdev         
    for i in range(n_iter):
        start = (i*batch_size)%train_set.shape[0] 
        x = train_set[start:start+batch_size]
        y = train_labels[start:start+batch_size]
        L = net.step(x,y)
        losses.append(L)
    for data in dev_set: 
        yhat = net(data) 
        yhats.append(torch.argmax(yhat,dim = 0))
    yhats = np.asarray(yhats)
    return losses,yhats,net