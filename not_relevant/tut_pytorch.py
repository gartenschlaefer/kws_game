# --
# some tutorial functions from the pytorch website
# not in interest of the project

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNetCifar(nn.Module):
  """
  Cifar Conv Network from pytorch tutorial
  """

  def __init__(self):
    """
    define network architecture
    """

    super().__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    """
    forward pass
    """
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x



class ConvNetTutorial(nn.Module):
  """
  Simple convolutional network adapted from the
  tutorial presented on the pytorch homepage
  """

  def __init__(self):
    """
    define neural network
    """

    # super: next method in line of method resolution order (MRO) 
    # from the base model nn.Module -> clears multiple inheritance issue
    super().__init__()

    # 1. conv layer
    self.conv1 = nn.Conv2d(1, 6, kernel_size=(3, 3), stride=(1, 1))

    # 2. conv layer
    self.conv2 = nn.Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1))

    # fully connected layers with affine transformations: y = Wx + b
    self.fc1 = nn.Linear(16 * 6 * 6, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)


  def forward(self, x):
    """
    forward pass
    """

    # max pooling of 1. conv layer
    x = F.max_pool2d( F.relu(self.conv1(x)), kernel_size=(2, 2) )

    # max pooling of 2. conv layer
    x = F.max_pool2d( F.relu(self.conv2(x)), kernel_size=(2, 2) )

    # flatten output from 2. conv layer
    x = x.view(-1, np.product(x.shape[1:]))

    # fully connected layers
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))

    # final fully connected layer
    x = self.fc3(x)

    return x


def run_tutorial_net(net):
  """
  From pytorch tutorial
  """

  # params
  params = list(net.parameters())

  # input : [nSamples x nChannels x Height x Width]
  x = torch.randn(1, 1, 32, 32)
  
  # output
  y = net(x)

  # zero gradient buffers of all params 
  net.zero_grad()

  # backprops with random gradients
  y.backward(torch.randn(1, 10))

  # print some infos
  print("net: ", net)
  print(len(params))
  print("params[0] shape: ", params[0].shape)
  print("out: \n", y)


  # --
  # compute Loss

  # generate new output
  y = net(x)

  # target
  t = torch.randn(10).view(1, -1)

  # MSE Loss
  criterion = nn.MSELoss()

  # compute loss
  loss = criterion(y, t)

  # print loss
  print("loss: ", loss)

  # gradients
  print("fn 1: ", loss.grad_fn)
  print("fn 2: ", loss.grad_fn.next_functions[0][0])
  print("fn 3: ", loss.grad_fn.next_functions[0][0].next_functions[0][0])


  # --
  # backprop

  # zero gradient buffers of all params
  net.zero_grad()

  print("before backprop: \n", net.conv1.bias.grad)

  # apply backprop
  loss.backward()

  print("after backprop: \n", net.conv1.bias.grad)


  # --
  # update the weights

  # learning rate
  lr = 0.01

  # go through all parameters
  for w in net.parameters():

    # update parameters: w <- w - t * g
    w.data.sub_(lr * w.grad.data)


  # --
  # using optimizers

  # create optimizer
  optimizer = torch.optim.SGD(net.parameters(), lr=lr)


  # training loop --

  # zero gradient buffer
  optimizer.zero_grad()

  # traverse data into network
  y = net(x)

  # loss
  loss = criterion(y, t)

  # backprop
  loss.backward()

  # optimizer update
  optimizer.step()


def train_cifar10(net, retrain=False):
  """
  training on cifar10 dataset
  """

  import torchvision
  import torchvision.transforms as transforms
  import os

  # transform init
  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  # training and test set
  trainset = torchvision.datasets.CIFAR10(root='./ignore/cifar10', train=True, download=True, transform=transform)
  testset = torchvision.datasets.CIFAR10(root='./ignore/cifar10', train=False, download=True, transform=transform)

  # loaders
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
  testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

  # classes
  classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

  # save parameter path
  param_path = './ignore/cifar10/cifar_net.pth'

  # number of epochs
  num_epochs = 2

  # learning rate
  lr = 0.01

  # check if param file exists
  if not os.path.exists(param_path) or retrain:

    # MSE Loss
    criterion = nn.CrossEntropyLoss()

    # create optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # epochs
    for epoch in range(num_epochs):

      # cumulated loss
      cum_loss = 0.0

      # fetch data samples
      for i, data in enumerate(trainloader, 0):

        # inputs and labels
        x, l = data

        print("x: ", x.shape)
        print("l: ", l.shape)

        # zero parameter gradients
        optimizer.zero_grad()

        # forward pass
        y = net(x)

        #print("y: ", y.shape)
        #print("l: ", l.shape)

        # loss
        loss = criterion(y, l)

        # backward
        loss.backward()

        # optimizer step
        optimizer.step()

        # statistics
        cum_loss += loss.item()

        # print every 2000 mini-batches
        if i % 2000 == 1999:

          # print
          print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, cum_loss / 2000))
          
          # zero cum loss
          cum_loss = 0.0

    print('Training finished')

    # save parameters of network
    torch.save(net.state_dict(), param_path)

  # load params from file
  else:

    # load
    net.load_state_dict(torch.load(param_path))


  # --
  # evaluation on test set

  # create an iterator
  data_iter = iter(testloader)

  # get next images in batch_sizes
  images, labels = data_iter.next()

  # classify
  y = net(images)

  # make prediction
  _, predicted = torch.max(y, 1)


  # show some images
  #cifar_imshow(torchvision.utils.make_grid(images))

  # print ground truth and predicted
  print("images: ", images.shape)
  print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
  print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))


  # --
  # evaluate whole dataset

  # metric init
  correct = 0
  total = 0

  # no gradients for eval
  with torch.no_grad():

    # load data
    for data in testloader:

      # extract sample
      images, labels = data

      # classify
      outputs = net(images)

      # prediction
      _, predicted = torch.max(outputs.data, 1)

      #print("predicted: ", predicted[0].item())
      #print("l: ", labels[0].item())

      # add total amount of prediction
      total += labels.size(0)

      # check if correctly predicted
      correct += (predicted == labels).sum().item()

  # plot accuracy
  print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


  # metric init
  class_correct = list(0. for i in range(10))
  class_total = list(0. for i in range(10))

  # no gradients for eval
  with torch.no_grad():

    for data in testloader:

      # extract sample
      images, labels = data

      # classify
      outputs = net(images)

      # prediction
      _, predicted = torch.max(outputs, 1)

      # class predicted labels that are correct
      c = (predicted == labels).squeeze()

      # for all batches
      for i in range(4):
        
        # update metrics
        class_correct[labels[i]] += c[i].item()
        class_total[labels[i]] += 1

  # print class dependent accuracies
  for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))


def cifar_imshow(img):
  """
  imshow for tutorial
  """

  # denormalize
  img = img / 2 + 0.5

  # plot image
  plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))

  plt.show()


if __name__ == '__main__':

  # net
  run_tutorial_net(ConvNetTutorial())

  # cifar tutorial
  #train_cifar10(ConvNetCifar(), retrain=False)