import torch
import torchvision
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import os
import json

class AlexNet(nn.Module):
    def __init__(self,num_class = 5):
        super().__init__()  # 继承nn.model类
        '''第一层卷积，卷积核大小为11*11，步距为4，输入通道为3，输出通道为96(feature map)'''
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0)
        '''第一层池化层，卷积核为3*3，步距为2'''
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(96,256,kernel_size=5,stride=1,padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv3 = nn.Conv2d(256,384,kernel_size=3,stride=1,padding=1)
        self.conv4 = nn.Conv2d(384,384,kernel_size=3,stride=1,padding=1)
        self.conv5 = nn.Conv2d(384,256,kernel_size=3,stride=1,padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.linear1 = nn.Linear(6*6*256,4096)
        self.linear2 = nn.Linear(4096, 4096)
        self.linear3 = nn.Linear(4096, 5)#花数据集，类别数为5
    def forward(self, x):
        out = torch.relu(self.conv1(x))
        out = self.pool1(out)
        out = torch.relu(self.conv2(out))
        out = self.pool2(out)
        out = torch.relu(self.conv3(out))
        out = torch.relu(self.conv4(out))
        out = torch.relu(self.conv5(out))
        out = self.pool3(out)

        out = out.reshape(-1, 6 * 6 * 256)  # flatten
        #out = torch.dropout(out,0.5,True)
        out = torch.relu(self.linear1(out))
        #out = torch.dropout(out,0.5,True)
        out = torch.relu(self.linear2(out))
        out = self.linear3(out)
        return out

'''定义参数'''
batch_size = 64
lr = 0.001
num_classes=5
total_epoch = 400
writer = SummaryWriter(log_dir='logs/log2/')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(227),  #随机剪裁成227*227的图像
                                 transforms.RandomHorizontalFlip(),  #一定概率水平翻转，默认0.5，（图像增强）
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((227, 227)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

data_root = os.path.abspath(os.path.join(os.getcwd(), "./"))  # get data root path
image_path = data_root + "/flower_data/"  # flower data set path
train_dataset = torchvision.datasets.ImageFolder(root=image_path + "/train",
                                     transform=data_transform["train"])
train_num = len(train_dataset)

validate_dataset = torchvision.datasets.ImageFolder(root=image_path + "/val",
                                        transform=data_transform["val"])
val_num = len(validate_dataset)

# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())

# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=4)

validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=4, shuffle=False)


def do_train():
    model = AlexNet(num_classes)
    model = model.cuda()
    '''设置损失函数'''
    criterion=nn.CrossEntropyLoss()
    '''设置优化器'''
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    '''开始训练'''
    total_step = len(train_loader)
    best_acc = 0.0
    for epoch in range(total_epoch):
        model.train()
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda()
            labels = labels.cuda()
            images = Variable(images)
            labels = Variable(labels)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss
            # if (i + 1) % 10 == 0:
            # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
            #       .format(epoch + 1, total_epoch, i + 1, total_step, loss.item()))
        writer.add_scalar("Train_Loss: ",total_loss/len(train_loader),epoch)
        writer.flush()

        model.eval()
        acc = 0.0
        with torch.no_grad():
            for val_data in validate_loader:
                val_images, val_labels = val_data
                outputs = model(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += (predict_y == val_labels.to(device)).sum().item()
            val_accurate = acc / val_num
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(model, './model/AlexNet_nodropout400.pth')
            print('Epoch [%d/%d] train_loss: %.3f  test_accuracy: %.3f' %
                  (epoch + 1,total_epoch,total_loss/len(train_loader), val_accurate))
            writer.add_scalar("Test_Accurate: ", val_accurate, epoch)
            writer.flush()
    #torch.save(model, './model/AlexNet_dropout400.pth')

def do_test():
    model = torch.load('./model/AlexNet_nodropout400.pth')
    model.eval()
    model = model.cuda()

    '''设置损失函数'''
    criterion=nn.CrossEntropyLoss()
    '''设置优化器'''
    #optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    correct = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(validate_loader):
            images = images.cuda()
            labels = labels.cuda()
            images = Variable(images)
            labels = Variable(labels)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            #获取预测结果的下标（即预测的数字）
            _,preds = torch.max(outputs,dim=1)
            #累计预测正确个数
            correct += torch.sum(preds == labels)
    print('accuracy rate: ',correct/len(validate_loader.dataset))

do_train()
#do_test()
writer.close()
