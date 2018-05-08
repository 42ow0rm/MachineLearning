import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np


# Hyper Parameters
num_epochs = 0
load_epoch = 30
batch_size = 4
learning_rate = 0.001
alg = "relu"

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=32)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=32)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        #x = F.sigmoid(self.fc1(x))
        #x = F.tanh(self.fc1(x))
        x = F.relu(self.fc1(x))

        #x = F.sigmoid(self.fc2(x))
        #x = F.tanh(self.fc2(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def train_network(step):
    net.train()
    epoch = 0
    for epoch in range(load_epoch, step):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    # Save the Trained Model
    torch.save(net, '_brain/' + alg + '_nn_CNN_c_' + str(epoch) + '.420')

def test_network():
    net.eval()
    class_correct = list(0. for i in range(1000))
    class_total = list(0. for i in range(1000))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            labels = labels.to(device)
            images = images.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

net = Net()
# load the last progress
if os.path.isfile('_brain/' + alg + '_nn_CNN_c_' + str(load_epoch) + '.420'):
    net = torch.load('_brain/' + alg + '_nn_CNN_c_' + str(load_epoch) + '.420')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# test foror GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
# Assume that we are on a CUDA machine, then this should print a CUDA device:
print(device)
net.to(device)

train_network(num_epochs + load_epoch)

test_network()

