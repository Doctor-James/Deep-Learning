import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn
true_w = torch.tensor([2, -3.4])
true_b = 4.2
#生成一个包含1000个样本的数据集，每个样本包含从标准正态分布中采样的2个特征。我们的合成数据集是一个矩阵 𝐗∈ℝ1000×2
#X是1*2矩阵，代表两个特征（类似于两个像素值），Y是一个标量，代表label
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

'''
散点图
d2l.set_figsize()
d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1);
d2l.plt.show()
'''

def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器。"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

#读取数据
batch_size = 10
data_iter = load_array((features, labels), batch_size)
next(iter(data_iter))

net = nn.Sequential(nn.Linear(2, 1)) # 调用API构建全连接神经网络，参数为输入维度，输出维度
'''
# Example of using Sequential
一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数。
model = nn.Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )
'''


'''初始化模型参数'''
#通过net[0]选择网络中的第一个图层
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss() #[计算均方误差使用的是MSELoss类，也称为平方 𝐿2 范数]。默认情况下，它返回所有样本损失的平均值

trainer = torch.optim.SGD(net.parameters(), lr=0.03)
num_epochs = 3

for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)