"""
adversarial networks
D - Discriminator
G - Generator
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from conv_nets import ConvBasics, ConvEncoder, ConvDecoder


class AdvBasics(ConvBasics):
  """
  Adversarial networks basic functions
  """

  def weights_init(self, module):
    """
    custom weights initialization called on netG and netD
    adapted form: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    """

    # get class name from modules
    cls_name = module.__class__.__name__

    # conv layer init
    if cls_name in ['Conv2d', 'ConvTranspose2d']:
      #print("{}: {} weights init".format(self.__class__.__name__, cls_name))
      nn.init.normal_(module.weight.data, 0.0, 0.02)

    # batch norm init
    elif cls_name.find('BatchNorm') != -1:
      #print("{} weights init".format(cls_name))
      nn.init.normal_(module.weight.data, 1.0, 0.02)
      nn.init.constant_(module.bias.data, 0)



class G_experimental(nn.Module, AdvBasics):
  """
  Generator for experimental purpose
  input: noise of size [n_latent]
  output: [batch x channels x m x f]
  m - features
  f - frames
  """

  def __init__(self, n_classes, data_size, n_latent=100):

    # parent init
    super().__init__()

    # arguments
    self.n_classes = n_classes
    self.data_size = data_size
    self.n_latent = n_latent

    # convolutional decoder
    self.conv_decoder = ConvDecoder(self.n_classes, self.data_size, n_latent=self.n_latent)

    # fully connected layers
    self.fc1 = nn.Linear(self.n_latent, 32)
    self.fc2 = nn.Linear(32, np.prod(self.conv_decoder.conv_in_dim))

    # last layer activation
    self.sigm = nn.Sigmoid()

    # init weights
    self.apply(self.weights_init)


  def forward(self, x):
    """
    forward pass
    """

    # fully connected layers
    x = self.fc1(x)
    x = F.relu(self.fc2(x))

    # reshape for conv layer
    x = torch.reshape(x, ((x.shape[0],) + self.conv_decoder.conv_in_dim))

    # conv decoder
    x = self.conv_decoder(x)

    # last layer activation
    x = self.sigm(x)

    return x



class D_experimental(nn.Module, AdvBasics):
  """
  Discriminator for experimental purpose
  input: [batch x channels x m x f]
  output: [1]
  m - features
  f - frames
  """

  def __init__(self, n_classes, data_size):

    # parent init
    super().__init__()

    # arguments
    self.n_classes = n_classes
    self.data_size = data_size

    # encoder model
    self.conv_encoder = ConvEncoder(self.n_classes, self.data_size)

    # fully connected layers
    self.fc1 = nn.Linear(np.prod(self.conv_encoder.conv_out_dim), 1)

    # dropout layer
    self.dropout_layer1 = nn.Dropout(p=0.5)

    # sigmoid
    self.sigm = nn.Sigmoid()

    # init weights
    self.apply(self.weights_init)


  def forward(self, x):
    """
    forward pass
    """

    # encoder model
    x = self.conv_encoder(x)

    # flatten output from conv layer
    x = x.view(-1, np.product(x.shape[1:]))

    # fully connected layer
    x = self.fc1(x)

    # last layer activation
    x = self.sigm(x)

    return x



if __name__ == '__main__':
  """
  main function
  """

  # latent variable
  n_latent = 100

  # noise
  noise = torch.randn(2, n_latent)

  # generate random sample
  x = torch.randn((1, 1, 39, 32))

  # create net
  G = G_experimental(n_classes=1, data_size=x.shape[1:], n_latent=n_latent)
  D = D_experimental(n_classes=1, data_size=x.shape[1:])

  # print some infos
  print("Net: ", D)
  print("Encoder: ", D.conv_encoder)

  # go through all modules
  for module in D.children():
    print("module cls: ", module.__class__.__name__)
    print("module: ", module)

  # test nets
  o_d = D(x)
  o_g = G(noise)

  # output
  print("d: ", o_d.shape)
  print("g: ", o_g.shape)
