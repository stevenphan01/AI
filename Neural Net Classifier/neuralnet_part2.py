import numpy as np
import torch

class NeuralNet(torch.nn.Module):
    def __init__(self, lrate,loss_fn,in_size, out_size, decay_weight):
        super(NeuralNet, self).__init__()
        #best one yet
        self.convLayers = torch.nn.Sequential(torch.nn.Conv2d(3,6,5),
                                        torch.nn.Dropout(),
                                        torch.nn.MaxPool2d(2,2),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv2d(6,16,5),
                                        torch.nn.Dropout(),                                             
                                        torch.nn.MaxPool2d(2,2))       
        self.fc = torch.nn.Sequential(torch.nn.Linear(16*5*5, 200),
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(200,out_size),
                                      torch.nn.ReLU()
                                      )
        self.lrate = lrate
        self.loss_fn = loss_fn
        self.optim = torch.optim.SGD(self.parameters(), lr=self.lrate, momentum=0.97, weight_decay=decay_weight)

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        x = self.convLayers(x)
        x = x.view(-1,x.shape[1]*x.shape[2]*x.shape[3])
        #print(x.shape)
        x = self.fc(x)
        return x 
    
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
    decay_weight = 1e-2 #1e-3 is the best so far
    net = NeuralNet(lrate, loss_fn, train_set.shape[1], 2, decay_weight)
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
        end = start + batch_size
        x = train_set[start:end]
        y = train_labels[start:end]
        L = net.step(x,y)
        losses.append(L) 
    with torch.no_grad():
        for data in dev_set: 
            yhat = net(data) 
            yhats.append(torch.argmax(yhat,dim = 1))
    return losses,yhats,net