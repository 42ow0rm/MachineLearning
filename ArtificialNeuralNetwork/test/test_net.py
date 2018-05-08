import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import torchvision
import random

from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
from os import listdir


#initiate class for Neural Network
class NeuralNet(nn.Module):
	def __init__(self):
		#needed 		
		super(NeuralNet, self).__init__()
		#configure all layers here
		self.lin1 = nn.Linear(10,10)
		self.lin2 = nn.Linear(10,10)
		
	def forward(self, x):
		#activation function commes here
		x = F.relu(self.lin1(x))
		x = self.lin2(x)
		return x
		
	def num_flat_features(self, x):
		size =x.size()[1:]
		num = 1
		for i in size:
			num *= i
		return num

	
#initiate the Neural Network		
netz = NeuralNet()
netz = netz.cuda() #for gc
print(netz)

#load the las progress
if os.path.isfile('test_net.420'):
	netz = torch.load('test_net.420')


#here comes the input
input = Variable(torch.randn(10,10))
input = input.cuda() #for gc
print(input)

#here you can get your output
out = netz(input)
print(out)

#heres come the criterion, to calctulate the errors
criterion = nn.MSELoss()

#set all gradient to zero
netz.zero_grad()

#save the neural network learn torch.save(object, file)
torch.save(netz, 'test_net.420')