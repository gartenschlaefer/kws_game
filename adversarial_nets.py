"""
adversarial networks
D - Discriminator
G - Generator
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdvBasics():
  """
  Convolutional networks Basic useful functions
  """

  def get_conv_layer_dimensions(self):
    """
    get convolutional layer dimensions upon kernel sizes and strides
    """

    # layer dimensions
    self.conv_layer_dim = []
    self.conv_layer_dim.append((self.n_features, self.n_frames))

    for i, (k, s) in enumerate(zip(self.kernel_sizes, self.strides)):
      self.conv_layer_dim.append((int((self.conv_layer_dim[i][0] - k[0]) / s[0] + 1), int((self.conv_layer_dim[i][1] - k[1]) / s[1] + 1)))

    print("conv layer dim: ", self.conv_layer_dim)


  def get_weights(self):
    """
    analyze weights of model interface
    """
    return None


  def weights_init(self, module):
    """
    custom weights initialization called on netG and netD
    adapted form: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    """

    # get class name from modules
    cls_name = module.__class__.__name__

    # conv layer inít
    if cls_name.find('Conv') != -1:
      print("{}: {} weights init".format(self.__class__.__name__, cls_name))
      torch.nn.init.normal_(module.weight.data, 0.0, 0.02)

    # batch norm init
    elif cls_name.find('BatchNorm') != -1:
      print("{} weights init".format(cls_name))
      torch.nn.init.normal_(module.weight.data, 1.0, 0.02)
      torch.nn.init.constant_(module.bias.data, 0)



class G_experimental(nn.Module, AdvBasics):
  """
  Generator for experimental purpose
  input: noise of size [n_latent]
  output: [batch x channels x m x f]
  m - features, e.g. (MFCC-39)
  f - frames, e.g. (32)
  """

  def __init__(self, n_classes, data_size, n_latent=100):

    # parent init
    super().__init__()

    # arguments
    self.n_classes = n_classes
    self.data_size = data_size
    self.n_latent = n_latent

    # extract input size [channel x features x frames]
    self.n_channels, self.n_features, self.n_frames = self.data_size

    # params (reversed order)
    self.n_feature_maps = [8, 4]
    self.kernel_sizes = [(self.n_features, 20), (1, 5)]
    self.strides = [(1, 5), (1, 1)]

    # get layer dimensions (reversed order for Generator)
    self.get_conv_layer_dimensions()

    # fully connected layers
    self.fc1 = nn.Linear(self.n_latent, 32)
    self.fc2 = nn.Linear(32, np.prod(self.conv_layer_dim[-1]) * self.n_feature_maps[1])

    # conv layer
    self.deconv1 = nn.ConvTranspose2d(in_channels=self.n_feature_maps[1], out_channels=self.n_feature_maps[0], kernel_size=self.kernel_sizes[1], stride=self.strides[1])
    #self.bn1 = nn.BatchNorm2d(self.n_feature_maps[0])

    self.deconv2 = nn.ConvTranspose2d(in_channels=self.n_feature_maps[0], out_channels=1, kernel_size=self.kernel_sizes[0], stride=self.strides[0])
    #self.bn2 = nn.BatchNorm2d(self.n_classes)

    # last layer activation
    self.tanh = nn.Tanh()
    #self.sigm = nn.Sigmoid()

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
    x = torch.reshape(x, ((x.shape[0], self.n_feature_maps[1]) + self.conv_layer_dim[-1]))

    # deconvolution
    x = self.deconv1(x)
    x = self.deconv2(x)

    # last layer activation
    x = self.tanh(x)
    #x = self.sigm(x)

    return x



class D_experimental(nn.Module, AdvBasics):
  """
  Discriminator for experimental purpose
  input: [batch x channels x m x f]
  output: [1]
  m - features (MFCC-39)
  f - frames (32)
  """

  def __init__(self, n_classes, data_size):

    # parent init
    super().__init__()

    # arguments
    self.n_classes = n_classes
    self.data_size = data_size

    # extract input size [channel x features x frames]
    self.n_channels, self.n_features, self.n_frames = self.data_size

    # params
    self.n_feature_maps = [8, 4]
    self.kernel_sizes = [(self.n_features, 20), (1, 5)]
    self.strides = [(1, 5), (1, 1)]

    # get layer dimensions
    self.get_conv_layer_dimensions()

    # conv layer
    self.conv1 = nn.Conv2d(self.n_channels, self.n_feature_maps[0], kernel_size=self.kernel_sizes[0], stride=self.strides[0])
    #self.bn1 = nn.BatchNorm2d(self.n_feature_maps[0])

    self.conv2 = nn.Conv2d(self.n_feature_maps[0], self.n_feature_maps[1], kernel_size=self.kernel_sizes[1], stride=self.strides[1])
    #self.bn2 = nn.BatchNorm2d(self.n_classes)

    # fully connected layers with affine transformations: y = Wx + b
    self.fc1 = nn.Linear(np.prod(self.conv_layer_dim[-1]) * self.n_feature_maps[-1], 1)

    # dropout layer
    self.dropout_layer1 = nn.Dropout(p=0.5)
    self.dropout_layer2 = nn.Dropout(p=0.2)

    # last layer activation
    self.sigm = nn.Sigmoid()

    # init weights
    self.apply(self.weights_init)


  def forward(self, x):
    """
    forward pass
    """

    # 1. conv layer
    x = self.conv1(x)
    #x = self.bn1(x)
    x = F.relu(x)
    #x = self.dropout_layer2(x)

    # 2. conv layer
    x = self.conv2(x)
    #x = self.bn2(x)
    x = F.relu(x)
    x = self.dropout_layer1(x)

    # flatten output from conv layer
    x = x.view(-1, np.product(x.shape[1:]))

    # fully connected layer
    x = self.fc1(x)
    #x = F.relu(x)
    #x = self.dropout_layer1(x)

    # 2. fully connected layer
    #x = self.fc2(x)

    # last layer activation
    x = self.sigm(x)

    return x


  def get_weights(self):
    """
    get weights of model
    """
    return {'conv1': self.conv1.weight.detach().cpu()}


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
