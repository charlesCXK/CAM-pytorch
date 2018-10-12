# --*-- encoding: utf-8 --*--

import pickle
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


from matplotlib import pyplot as plt
from matplotlib import cm as CM
from matplotlib import axes
from torch.autograd import Variable
from torch import optim
from PIL import Image


''' define a network structure '''
cfg = [64, 'M', 128, 'M', 256, 256]    # 自己定义一个网络
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class mynet(nn.Module):
    def __init__(self):
        super(mynet, self).__init__()
        self.features = self._make_layers(cfg)
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.linear = torch.nn.Sequential(
            nn.Linear(256, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
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
        return nn.Sequential(*layers)
    def forward(self, x):
        x1 = self.features(x)
        x_avg = self.avgpool(x1)
        x = x_avg.view(-1, 256)
        x = self.linear(x)
        return x, x_avg, x1
net = mynet()

# if I have a GPU, use it.
net.to(device)
# transform GPU model into CPU network
net.load_state_dict(torch.load('params.pkl', map_location=lambda storage, loc: storage))



""" Load the train dataset """
# 数据预处理，转化为 Tensor
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='', train=True, download=False, transform=transform)
trainloader=torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=False, num_workers=2)

# only try the first batch
for batch, data in enumerate(trainloader, 0):
	inputs, labels = data
	outputs, avg_weight, features = net(inputs)

	for i in range(len(inputs)):
		fea = features[i]
		avg = avg_weight[i]
		for j in range(256):
			# print(fea[j].shape)
			# print(avg[j][0][0])
			fea[j] = fea[j]*avg[j][0][0]	
		sum_fea = torch.sum(fea, 0)		# sum the 256 feature maps	
		sum_fea = (sum_fea*255/sum_fea.max()).long().numpy()
		plt.imshow(sum_fea, cmap="jet")
		plt.show()
		sum_fea = np.tile(sum_fea, (3, 1, 1))
		sum_fea = np.swapaxes(sum_fea, 0, 1)
		sum_fea = np.swapaxes(sum_fea, 1, 2)

		img = Image.fromarray(sum_fea.astype('uint8')).resize((224, 224))

		raw_img = (inputs[i]*255).long().numpy()
		raw_img = np.swapaxes(raw_img, 0, 1)
		raw_img = np.swapaxes(raw_img, 1, 2)
		raw_img = Image.fromarray(raw_img.astype('uint8')).resize((224, 224))
		
		final_img = np.concatenate((raw_img, img), axis=1)
		final_img = Image.fromarray(final_img.astype('uint8'))
		final_img.save('res/{}.png'.format(i), 'PNG')
		break
	break