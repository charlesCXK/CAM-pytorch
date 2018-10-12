
# coding: utf-8
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

from torch.autograd import Variable
from torch import optim
from PIL import Image


""" Load the train dataset """
# 数据预处理，转化为 Tensor
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='', train=True, download=False, transform=transform)
trainloader=torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=4)


""" Load the test dataset """
# 数据预处理，转化为 Tensor
transform = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.CIFAR10(root='', train=False, download=False, transform=transform)
testloader=torch.utils.data.DataLoader(testset, batch_size=16, shuffle=True, num_workers=4)


classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')


''' define a network structure '''
cfg = [64, 'M', 128, 'M', 256, 256]    # 自己定义一个网络
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class mynet(nn.Module):
    def __init__(self):
        super(mynet, self).__init__()
        self.features = self._make_layers(cfg)
        self.linear = torch.nn.Sequential(
            nn.Linear(256, 10),
        )
        
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=8, stride=1)]
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256)
#         print(x.shape)

        x = self.linear(x)
        return x
net = mynet()

# if I have a GPU, use it.
net.to(device)


''' train this model '''
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(),lr=0.004,momentum=0.9)
epoch_num = 30

for epoch in range(epoch_num):
    print('-------- epoch {} --------'.format(epoch))
    running_loss = 0
    for i, data in enumerate(trainloader,0):
        inputs, labels = data
        inputs,labels = Variable(inputs).to(device), Variable(labels).to(device)
        
        optimizer.zero_grad()    # 梯度清零
        outputs = net(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.data[0]
        if i%200 == 0:
            print(loss)

''' test the dataset '''
tot_case, correct_case = 0, 0
for i, data in enumerate(testloader,0):
    inputs, labels = data
    inputs,labels = Variable(inputs).to(device), Variable(labels).to(device)
    outputs = net(inputs)
    for i in range(len(outputs)):
        out, label = outputs[i], labels[i].long()
        out_max = torch.max(out, 0)[1].long()
        tot_case += 1
        if torch.equal(out_max, label) == True:
            correct_case += 1
print('accuracy is {}%'.format(correct_case/tot_case*100))


''' save the model '''
torch.save(net.state_dict(), 'params.pkl')

