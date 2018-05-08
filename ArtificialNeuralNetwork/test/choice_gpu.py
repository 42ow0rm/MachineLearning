import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import random

from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
from os import listdir

# path the images of the dataset to train
path_train = 'res/cat_dog/train/'

# path the images of the dataset to test
path_test = 'res/cat_dog/test/'

# epoch to be loaded, start with 1
load_epoch = 1

# step is the count of cycle  y=-1 for test
step = 100


# ------------------------------------------------------------------------------------------
# initiate class for Neural Network
class NeuralNet(nn.Module):
    def __init__(self):
        # needed
        super(NeuralNet, self).__init__()
        # configure all layers here
        # the layers can be experimented
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)  # kernel_size can be experimented
        self.conv2 = nn.Conv2d(6, 12, kernel_size=5)  # Conv2D(x,y): x is the in layer and y the out layer
        self.conv3 = nn.Conv2d(12, 18, kernel_size=5)  # layer1.y = layer2.x
        self.conv4 = nn.Conv2d(18, 24, kernel_size=5)
        self.conv5 = nn.Conv2d(24, 30, kernel_size=5)

        # get the amount of layer by uncomment bellow
        self.fc1 = nn.Linear(480, 10)
        self.fc2 = nn.Linear(10, 2)  # Linear(x,y): y is the amount of items $nbr_items

    def forward(self, x):
        # activation function commes here
        # free to choose the learning steps, functions that here to be used
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        # x = F.relu(x)
        # x = F.sigmoid(x)
        x = F.tanh(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        # x = F.relu(x)
        # x = F.sigmoid(x)
        x = F.tanh(x)

        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        # x = F.relu(x)
        # x = F.sigmoid(x)
        x = F.tanh(x)

        x = self.conv4(x)
        x = F.max_pool2d(x, 2)
        # x = F.relu(x)
        # x = F.sigmoid(x)
        x = F.tanh(x)

        x = self.conv5(x)
        x = F.max_pool2d(x, 2)
        # x = F.relu(x)
        # x = F.sigmoid(x)
        x = F.tanh(x)

        # show how many output we have for the next layers
        print(x.size())
        exit()

        # create a view -1 for ignoring first columns (batch_id))
        x = x.view(-1, 480)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.sigmoid(x)


# ------------------------------------------------------------------------------------------
# initiate datasets

# set a normalize transformation
# the values can be experimented
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# set a list of transformations to do with the images
transforms = transforms.Compose([transforms.Resize(128), transforms.CenterCrop(128), transforms.ToTensor(),
                                 normalize])

train_data_list = []  # list of the images in tensors
target_list = []
train_data = []  # list of the stacked tensors(images)
files = listdir(path_train)

# for loop to mix all data in list
for i in range(len(listdir(path_train))):
    f = random.choice(files)  # choose a random element from the list
    files.remove(f)  # remove the file from the list so that it came only 1 time

    img = Image.open(path_train + f)  # load the image
    img_tensor = transforms(img)  # convert the image in tensors

    train_data_list.append(img_tensor)  # add the tensor to the list

    # here we need to say how many different items we have $nbr_items
    # variant I:
    # isItem1 = 1 if 'item1' in f else 0
    # isItem2 = 1 if 'item2' in f else 0
    # isItem3 = 1 if 'item3' in f else 0
    # ...
    #
    # variant II:
    # target = 1 for item1, 2 for item2, ...
    #
    isItem1 = 1 if 'cat' in f else 0
    isItem2 = 1 if 'dog' in f else 0
    target = [isItem1, isItem2]
    target_list.append(target)

    # to make the list not too long we make cuts of for ex. 64 items
    if len(train_data_list) >= 256:
        train_data.append((torch.stack(train_data_list), target_list))
        train_data_list = []
        print('Loaded batch ', len(train_data), 'of', int(len(listdir(path_train))))
        print('Percentage done: ', 100 * len(train_data) / int(len(listdir(path_train))))
        #break

    if len(train_data) >= 1:
        break

# ------------------------------------------------------------------------------------------
# initiate the Neural Network
net = NeuralNet()
net = net.cuda() #for gc
# print(net)

# load the last progress
if os.path.isfile('_brain/nn_choice_c_' + str(load_epoch) + '.420'):
    net = torch.load('_brain/nn_choice_c_' + str(load_epoch) + '.420')

# define the optimizer, lr is the learn rate
optimizer = optim.Adam(net.parameters(), lr=0.01)


# define the train function
def train(epoch):
    # set the network in train mode
    net.train()
    batch_id = 0

    for data, target in train_data:
        # convert data in tensors
        target = torch.Tensor(target).cuda()
        data = data.cuda()

        # convert tensors in Variables
        data = Variable(data)
        target = Variable(target)

        # set optimizer gradient to zero
        optimizer.zero_grad()

        # getting output
        out = net(data)

        # heres come the criterion, to calculate the errors
        criterion = F.smooth_l1_loss
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        # print function to see the progress of the the training
        print('Train cycle: {} [{}/{}[  ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_id + len(data), len(train_data),
                                                                         100. * batch_id / len(train_data),
                                                                         loss.data[0]))

        batch_id += 1


# if ((batch_id % 100) == 0):
# save the neural network learn torch.save(object, file)
# torch.save(net, '_brain/nn_choice_c_' + str(epoch) + '.420')


# define the test function
def test():
    # set the network in evaluation mode
    net.eval()
    files = listdir(path_train)
    f = random.choice(files)
    img = Image.open(path_train + f)
    img_eval_tensor = transforms(img)
    img_eval_tensor.unsqueeze_(0)
    data = Variable(img_eval_tensor)
    out = net(data)
    if (out.data.max(1, keepdim=True)[1] == 1):
        print('Dog')
    else:
        print('Cat')
    img.show()


# pause, waiting for user interaction
# x = input('')

# included to simplify tests
def train_network(step):
    if (step > -1):
        # main function of the network
        for epoch in range(start_epoch, (start_epoch + step)):
            train(epoch)

        # save the network
        torch.save(net, '_brain/nn_choice_c_' + str(epoch) + '.420')


# resume epoch
start_epoch = load_epoch

train_network(step)

test()
