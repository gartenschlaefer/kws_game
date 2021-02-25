"""
convolutional neural network architectures
partly adapted from pytorch tutorial

References:
[Sainath 2015] - Tara N. Sainath and Carolina Parada. Convolutional neural networks for small-footprint key-word spotting. InINTERSPEECH, 2015
"""

import numpy as np

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
    input: [batch x channels x m x f]
    m - features (MFCC)
    f - frames
    """

    # parent init
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

    # dropout layer
    self.dropout_layer1 = nn.Dropout(p=0.2)
    self.dropout_layer2 = nn.Dropout(p=0.5)

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
    x = self.dropout_layer1(x)

    # 2. fully connected layers [1 x 128]
    x = F.relu(self.fc2(x))
    x = self.dropout_layer2(x)

    # Softmax layer [1 x n_classes]
    x = self.softmax(self.fc3(x))

    return x



class ConvNetFstride4(nn.Module):
  """
  Conv Net architecture with limited multipliers 
  presented in [Sainath 2015] - cnn-one-fstride4
  """

  def __init__(self, n_classes, data_size):
    """
    define neural network architecture
    input: [batch x channels x m x f]
    m - features (MFCC)
    f - frames
    """

    # parent init
    super().__init__()

    # arguments
    self.n_classes = n_classes
    self.data_size = data_size

    # extract input size [channel x features x frames]
    self.n_channels, self.n_features, self.n_frames = self.data_size

    # params
    self.n_feature_maps = 54
    self.kernel_size = (8, self.n_frames)
    self.stride = (4, 1)

    # dimensions after conv
    self.feature_strides = int((self.n_features - self.kernel_size[0]) / self.stride[0] + 1)
    self.frame_strides = int((self.n_frames - self.kernel_size[1]) / self.stride[1] + 1)

    print("feature_strides: ", self.feature_strides)
    print("frame_strides: ", self.frame_strides)

    # conv layer
    self.conv = nn.Conv2d(self.n_channels, self.n_feature_maps, kernel_size=self.kernel_size, stride=self.stride)

    # fully connected layers with affine transformations: y = Wx + b
    self.fc1 = nn.Linear(self.n_feature_maps * self.feature_strides * self.frame_strides, 32)
    self.fc2 = nn.Linear(32, 128)
    self.fc3 = nn.Linear(128, 128)
    self.fc4 = nn.Linear(128, self.n_classes)

    # dropout layer
    self.dropout_layer1 = nn.Dropout(p=0.2)
    self.dropout_layer2 = nn.Dropout(p=0.5)

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
    x = self.dropout_layer1(x)

    # 2. fully connected layers [1 x 128]
    x = F.relu(self.fc2(x))
    x = self.dropout_layer2(x)

    # 3. fully connected layers [1 x 128]
    x = F.relu(self.fc3(x))
    x = self.dropout_layer2(x)

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

  # generate random sample
  x = torch.randn((1, 1, 39, 32))

  # create net
  net = ConvNetFstride4(n_classes=5, data_size=x.shape[1:])

  # test net
  o = net(x)

  # print some infos
  print("\nx: ", x.shape)
  print("Net: ", net)
  print("o: ", o)