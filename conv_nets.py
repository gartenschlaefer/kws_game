"""
convolutional neural network architectures
partly adapted from pytorch tutorial
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet1(nn.Module):
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
    forward propagation through network
    """

    # max pooling of 1. conv layer
    x = F.max_pool2d( F.relu(self.conv1(x)), kernel_size=(2, 2) )

    # max pooling of 2. conv layer
    x = F.max_pool2d( F.relu(self.conv2(x)), kernel_size=(2, 2) )

    # flatten output from 2. conv layer
    x = x.view(-1, self.num_flat_features(x))

    # fully connected layers
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))

    # final fully connected layer
    x = self.fc3(x)

    return x


  def num_flat_features(self, x):
    """
    get number of flat features without batch dimension
    """

    return np.product(x.shape[1:])



def run_tutorial_net():
  """
  From pytorch tutorial
  """

  # convolutional network architecture 1
  net = ConvNet1()

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


if __name__ == '__main__':
  """
  main function
  """

  # run the tutorial net
  run_tutorial_net()
