"""
convolutional neural network architectures
partly adapted from pytorch tutorial

References:
[Sainath 2015] - Tara N. Sainath and Carolina Parada. Convolutional neural networks for small-footprint key-word spotting. InINTERSPEECH, 2015
"""

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNetTrad(nn.Module):
  """
  Traditional Conv Net architecture 
  presented in [Sainath 2015] - cnn-trad-fpool3
  """

  def __init__(self, n_classes):
    """
    define neural network architecture
    input: [m x f]
    m - features (MFCC)
    f - frames
    """

    # MRO check
    super().__init__()

    # 1. conv layer
    self.conv1 = nn.Conv2d(1, 64, kernel_size=(8, 20), stride=(1, 1))

    # max pool layer
    self.pool = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))

    # 2. conv layer
    self.conv2 = nn.Conv2d(64, 64, kernel_size=(4, 10), stride=(1, 1))

    # fully connected layers with affine transformations: y = Wx + b
    self.fc1 = nn.Linear(1280, 32)
    self.fc2 = nn.Linear(32, 128)
    self.fc3 = nn.Linear(128, n_classes)

    # softmax layer
    self.softmax = nn.Softmax(dim=1)


  def forward(self, x):
    """
    forward pass
    """

    # 1. conv layer [1 x 64 x 32 x 13]
    x = F.relu(self.conv1(x))

    # max pooling layer [1 x 64 x 8 x 13]
    x = self.pool(x)

    # 2. conv layer [1 x 64 x 5 x 4]
    x = F.relu(self.conv2(x))

    # flatten output from 2. conv layer [1 x 1280]
    x = x.view(-1, np.product(x.shape[1:]))

    # 1. fully connected layers [1 x 32]
    x = self.fc1(x)

    # 2. fully connected layers [1 x 128]
    x = F.relu(self.fc2(x))

    # Softmax layer [1 x n_classes]
    x = self.softmax(self.fc3(x))

    return x



class ConvNetFstride4(nn.Module):
  """
  Conv Net architecture with limited multipliers 
  presented in [Sainath 2015] - cnn-one-fstride4
  """

  def __init__(self, n_classes):
    """
    define neural network architecture
    input: [m x f]
    m - features (MFCC)
    f - frames
    """

    # MRO check
    super().__init__()

    # conv layer
    self.conv = nn.Conv2d(1, 54, kernel_size=(8, 32), stride=(4, 1))

    # fully connected layers with affine transformations: y = Wx + b
    self.fc1 = nn.Linear(432, 32)
    self.fc2 = nn.Linear(32, 128)
    self.fc3 = nn.Linear(128, 128)
    self.fc4 = nn.Linear(128, n_classes)

    # softmax layer
    self.softmax = nn.Softmax(dim=1)


  def forward(self, x):
    """
    forward pass
    """

    # 1. conv layer [1 x 54 x 8 x 1]
    x = F.relu(self.conv(x))

    # flatten output from conv layer [1 x 432]
    x = x.view(-1, np.product(x.shape[1:]))

    # 1. fully connected layers [1 x 32]
    x = self.fc1(x)

    # 2. fully connected layers [1 x 128]
    x = F.relu(self.fc2(x))

    # 3. fully connected layers [1 x 128]
    x = F.relu(self.fc3(x))

    # Softmax layer [1 x n_classes]
    x = self.softmax(self.fc4(x))

    return x



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



if __name__ == '__main__':
  """
  main function
  """

  # tutorial stuff
  from tutorial import run_tutorial_net, train_cifar10

  # run the tutorial net
  run_tutorial_net(ConvNetTutorial())

  # training example
  train_cifar10(ConvNetCifar(), retrain=False)
