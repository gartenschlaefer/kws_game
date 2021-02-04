"""
adversarial networks
D - Discriminator
G - Generator
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class G_experimental(nn.Module):
  """
  Generator for experimental purpose
  input: noise of size [n_latent]
  output: [batch x channels x m x f]
  m - features (MFCC-39)
  f - frames (32)
  """

  def __init__(self, n_latent=100):

    # parent init
    super().__init__()

    # latent space size
    self.n_latent = n_latent

    # conv layer
    #self.deconv = nn.Conv2d(1, 54, kernel_size=(8, 32), stride=(4, 1))

    # fully connected layers with affine transformations: y = Wx + b
    self.fc1 = nn.Linear(n_latent, 128)
    self.fc2 = nn.Linear(128, 128)
    self.fc3 = nn.Linear(128, 32)
    self.fc4 = nn.Linear(32, 432)

    # deconv layer
    #self.deconv = nn.ConvTranspose2d(in_channels=54, out_channels=1, kernel_size=(8, 32), stride=(4, 1), padding=0, bias=False)
    self.deconv = nn.ConvTranspose2d(in_channels=54, out_channels=1, kernel_size=(11, 32), stride=(4, 1), padding=0, bias=False)

    # softmax layer
    self.tanh = nn.Tanh()


  def forward(self, x):
    """
    forward pass
    """

    # linear layers

    # 1. fully connected layers [1 x 128]
    x = self.fc1(x)

    # 2. fully connected layers [1 x 128]
    x = F.relu(self.fc2(x))

    # 3. fully connected layers [1 x 32]
    x = F.relu(self.fc3(x))

    # 4. fully connected layers [1 x 432]
    x = F.relu(self.fc4(x))

    #[1 x 54 x 8 x 1]
    x = torch.reshape(x, (-1, 54, 8, 1))

    # unsqueeze [1 x 1 x 1 x 432]
    #x = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(x, 0), 0), 0)

    # deconvolution output_size=(1, 1, 39, 32)
    x = self.deconv(x)

    return x



class D_experimental(nn.Module):
  """
  Discriminator for experimental purpose
  input: [batch x channels x m x f]
  output: [1]
  m - features (MFCC-39)
  f - frames (32)
  """

  def __init__(self):

    # parent init
    super().__init__()

    # conv layer
    self.conv = nn.Conv2d(1, 54, kernel_size=(8, 32), stride=(4, 1))

    # fully connected layers with affine transformations: y = Wx + b
    self.fc1 = nn.Linear(432, 32)
    self.fc2 = nn.Linear(32, 128)
    self.fc3 = nn.Linear(128, 128)
    self.fc4 = nn.Linear(128, 1)

    # softmax layer
    self.sigm = nn.Sigmoid()


  def forward(self, x):
    """
    forward pass
    """

    # 1. conv layer [1, 1, 39, 32] -> [1 x 54 x 8 x 1]
    x = F.relu(self.conv(x))

    # flatten output from conv layer [1 x 432]
    x = x.view(-1, np.product(x.shape[1:]))

    # 1. fully connected layers [1 x 32]
    x = self.fc1(x)

    # 2. fully connected layers [1 x 128]
    x = F.relu(self.fc2(x))

    # 3. fully connected layers [1 x 128]
    x = F.relu(self.fc3(x))

    # 4. fully connected + sigmoid
    x = self.sigm(self.fc4(x))

    return x


if __name__ == '__main__':
  """
  main function
  """

  # latent variable
  n_latent = 100

  # noise
  noise = torch.randn(2, n_latent)

  # create net
  G = G_experimental(n_latent=n_latent)
  D = D_experimental()

  # print some infos
  print("Net: ", D)

  # generate random sample
  x = torch.randn((1, 1, 39, 32))

  print("\nx: ", x.shape)

  # test nets
  o_d = D(x)
  o_g = G(noise)


  # output
  print("d: ", o_d.shape)
  print("g: ", o_g.shape)
