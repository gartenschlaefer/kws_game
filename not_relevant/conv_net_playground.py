"""
playground for conv nets
"""

import torch
import torch.nn as nn
import numpy as np

if __name__ == '__main__':
  """
  main function
  """

  x = torch.zeros(1, 1, 3, 4)
  x[0, 0, 1, 1] = 1
  x[0, 0, 1, 0] = 1
  print("x: ", x)
  m = nn.Conv2d(1, 2, kernel_size=(3, 3), stride=(1, 1), bias=False)
  print("w: ", m.weight.shape)
  m.weight[0, 0] = 1.
  m.weight[1, 0] = 1.
  print("w: ", m.weight)
  print("b: ", m.bias)
  o = m(x)

  print("out: ", o)