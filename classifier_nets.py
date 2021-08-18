"""
classifier networks
"""

import numpy as np

import torch
import torch.nn as nn


class ClassifierNetFc1(nn.Module):
  """
  classifier network with one fully connected layer
  """

  def __init__(self, input_dim, output_dim):

    # parent init
    super().__init__()

    # fc dims
    self.fc_layer_dims = [(input_dim, output_dim)]

    # structure
    self.fc1 = nn.Linear(input_dim, output_dim)
    self.softmax =  nn.Softmax(dim=1)


  def forward(self, x):
    """
    forward pass
    """
    return self.softmax(self.fc1(x))



class ClassifierNetFc3(nn.Module):
  """
  classifier network with three fully connected layers
  """

  def __init__(self, input_dim, output_dim, dropout_enabled=True):

    # parent init
    super().__init__()

    # fc dims
    self.fc_layer_dims = [(input_dim, 64), (64, 32), (32, output_dim)]

    # structure
    self.fc1 = nn.Linear(input_dim, 64)
    self.fc2 = nn.Linear(64, 32)
    self.fc3 = nn.Linear(32, output_dim)
    self.dropout_layer = nn.Dropout(p=0.5) if dropout_enabled else nn.Identity()
    self.softmax = nn.Softmax(dim=1)


  def forward(self, x):
    """
    forward pass
    """
    x = torch.relu(self.fc1(x))
    x = self.dropout_layer(torch.relu(self.fc2(x)))
    return self.softmax(self.fc3(x))


if __name__ == '__main__':
  """
  main function
  """

  # create net
  net = ClassifierNetFc3(input_dim=5, output_dim=1, dropout_enabled=False)

  # print net
  print("Net: ", net)