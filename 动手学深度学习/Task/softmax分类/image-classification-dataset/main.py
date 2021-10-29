import load
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = load.load_data_fashion_mnist(batch_size)

# nn.Flatten()：任何维度tensor变成2D tensor
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)

net.apply(init_weights)

loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(),lr=0.01)

num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)