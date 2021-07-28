"""
adversarial networks with notations: D - Discriminator, G - Generator
"""

import numpy as np

import torch
import torch.nn as nn

from conv_nets import ConvBasics


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



class Adv_G_Experimental(nn.Module, AdvBasics):
  """
  adv generator jim label network
  """

  def __init__(self, n_classes, data_size, n_latent=100, is_last_activation_sigmoid=True):

    # parent init
    super().__init__()
    AdvBasics.__init__(self, n_classes, data_size)

    # arguments
    self.n_classes = n_classes
    self.data_size = data_size
    self.n_latent = n_latent

    # conv params (not transposed)
    self.feature_maps = [(self.n_channels, 8), (8, 8)]
    self.kernel_sizes = [(self.n_features, 20), (1, 5)]
    self.strides = [(1, 1), (1, 1)]
    self.padding = [(0, 0), (0, 0)]

    # dimensions
    self.conv_layer_dim = self.get_conv_layer_dimensions(input_dim=(self.n_features, self.n_frames), kernel_sizes=self.kernel_sizes, strides=self.strides, padding=self.padding)
    self.conv_in_dim = ((self.feature_maps[-1][1],) + self.conv_layer_dim[-1])
    self.conv_out_dim = self.data_size

    # fully connected
    self.fc1 = nn.Linear(self.n_latent, np.prod(self.conv_in_dim))

    # create deconv layers
    self.deconv_layer0 = nn.ConvTranspose2d(self.feature_maps[1][1], self.feature_maps[1][0], kernel_size=self.kernel_sizes[1], stride=self.strides[1], bias=False)
    self.deconv_layer1 = nn.ConvTranspose2d(self.feature_maps[0][1], self.feature_maps[0][0], kernel_size=self.kernel_sizes[0], stride=self.strides[0], bias=False)

    # last activation
    self.last_activation = nn.Sigmoid() if is_last_activation_sigmoid else nn.Identity()

    # init weights
    self.apply(self.weights_init)


  def forward(self, x):
    """
    forward pass
    """

    # fully connected layers
    x = self.fc1(x)

    # reshape for conv layer
    x = torch.reshape(x, ((x.shape[0],) + self.conv_in_dim))

    # conv decoder
    x = torch.relu(self.deconv_layer0(x))
    x = self.deconv_layer1(x)

    # last activation
    x = self.last_activation(x)

    return x



class Adv_D_Experimental(nn.Module, AdvBasics):
  """
  adv generator jim label network
  """

  def __init__(self, n_classes, data_size):

    # parent init
    super().__init__()
    AdvBasics.__init__(self, n_classes, data_size)

    # arguments
    self.n_classes = n_classes
    self.data_size = data_size

    # conv params
    self.feature_maps = [(self.n_channels, 8), (8, 8)]
    self.kernel_sizes = [(self.n_features, 20), (1, 5)]
    self.strides = [(1, 1), (1, 1)]
    self.padding = [(0, 0), (0, 0)]

    # get conv layer dimensions
    self.conv_layer_dim = self.get_conv_layer_dimensions((self.n_features, self.n_frames), self.kernel_sizes, self.strides, self.padding)
    self.conv_in_dim = self.data_size
    self.conv_out_dim = ((self.feature_maps[-1][1],) + self.conv_layer_dim[-1])

    # conv layer
    self.conv_layer0 = nn.Conv2d(self.feature_maps[0][0], self.feature_maps[0][1], kernel_size=self.kernel_sizes[0], stride=self.strides[0], bias=False)
    self.conv_layer1 = nn.Conv2d(self.feature_maps[1][0], self.feature_maps[1][1], kernel_size=self.kernel_sizes[1], stride=self.strides[1], bias=False)

    # fully connected
    self.fc1 = nn.Linear(np.prod(self.conv_out_dim), 1)

    # init weights
    self.apply(self.weights_init)


  def forward(self, x):
    """
    forward pass
    """

    # conv layers
    x = torch.relu(self.conv_layer0(x))
    x = torch.relu(self.conv_layer1(x))

    # flatten
    x = x.view(-1, np.product(x.shape[1:]))

    # fully connected layers
    x = self.fc1(x)

    # sigmoid
    x = torch.sigmoid(x)

    return x



class Adv_G_Jim(nn.Module, AdvBasics):
  """
  adv generator jim label network
  """

  def __init__(self, n_classes, data_size, n_latent=100, n_feature_maps_l0=48, is_last_activation_sigmoid=True):

    # parent init
    super().__init__()
    AdvBasics.__init__(self, n_classes, data_size)

    # arguments
    self.n_classes = n_classes
    self.data_size = data_size
    self.n_latent = n_latent
    self.n_feature_maps_l0 = n_feature_maps_l0

    # conv params (not transposed)
    self.feature_maps = [(self.n_channels, self.n_feature_maps_l0), (self.n_feature_maps_l0, 8)]
    self.kernel_sizes = [(self.n_features, 20), (1, 5)]
    self.strides = [(1, 1), (1, 1)]
    self.padding = [(0, 0), (0, 0)]

    # dimensions
    self.conv_layer_dim = self.get_conv_layer_dimensions(input_dim=(self.n_features, self.n_frames), kernel_sizes=self.kernel_sizes, strides=self.strides, padding=self.padding)
    self.conv_in_dim = ((self.feature_maps[-1][1],) + self.conv_layer_dim[-1])
    self.conv_out_dim = self.data_size

    # create deconv layers
    self.deconv_layer0 = nn.ConvTranspose2d(self.feature_maps[1][1], self.feature_maps[1][0], kernel_size=self.kernel_sizes[1], stride=self.strides[1], bias=False)
    self.deconv_layer1 = nn.ConvTranspose2d(self.feature_maps[0][1], self.feature_maps[0][0], kernel_size=self.kernel_sizes[0], stride=self.strides[0], bias=False)

    # fully connected
    self.fc1 = nn.Linear(self.n_latent, np.prod(self.conv_in_dim))

    # last activation
    self.last_activation = nn.Sigmoid() if is_last_activation_sigmoid else nn.Identity()

    # init weights
    self.apply(self.weights_init)


  def forward(self, x):
    """
    forward pass
    """

    # fully connected layers
    x = self.fc1(x)

    # reshape for conv layer
    x = torch.reshape(x, ((x.shape[0],) + self.conv_in_dim))

    # conv decoder
    x = torch.relu(self.deconv_layer0(x))
    x = self.deconv_layer1(x)

    # last activation
    x = self.last_activation(x)

    return x



class Adv_D_Jim(nn.Module, AdvBasics):
  """
  adv generator jim label network
  """

  def __init__(self, n_classes, data_size, n_feature_maps_l0=48):

    # parent init
    super().__init__()
    AdvBasics.__init__(self, n_classes, data_size)

    # arguments
    self.n_classes = n_classes
    self.data_size = data_size
    self.n_feature_maps_l0 = n_feature_maps_l0

    # conv params
    self.feature_maps = [(self.n_channels, self.n_feature_maps_l0), (self.n_feature_maps_l0, 8)]
    self.kernel_sizes = [(self.n_features, 20), (1, 5)]
    self.strides = [(1, 1), (1, 1)]
    self.padding = [(0, 0), (0, 0)]

    # get conv layer dimensions
    self.conv_layer_dim = self.get_conv_layer_dimensions((self.n_features, self.n_frames), self.kernel_sizes, self.strides, self.padding)
    self.conv_in_dim = self.data_size
    self.conv_out_dim = ((self.feature_maps[-1][1],) + self.conv_layer_dim[-1])

    # conv layer
    self.conv_layer0 = nn.Conv2d(self.feature_maps[0][0], self.feature_maps[0][1], kernel_size=self.kernel_sizes[0], stride=self.strides[0], bias=False)
    self.conv_layer1 = nn.Conv2d(self.feature_maps[1][0], self.feature_maps[1][1], kernel_size=self.kernel_sizes[1], stride=self.strides[1], bias=False)

    # fully connected
    self.fc1 = nn.Linear(np.prod(self.conv_out_dim), 1)

    # init weights
    self.apply(self.weights_init)


  def forward(self, x):
    """
    forward pass
    """

    # conv layers
    x = torch.relu(self.conv_layer0(x))
    x = torch.relu(self.conv_layer1(x))

    # flatten
    x = x.view(-1, np.product(x.shape[1:]))

    # fully connected layers
    x = self.fc1(x)

    # sigmoid
    x = torch.sigmoid(x)

    return x



if __name__ == '__main__':
  """
  main function
  """

  # latent var
  n_latent = 100

  # generate random sample
  x = torch.randn((1, 1, 39, 32))

  # create net
  G = Adv_G_Experimental(n_classes=1, data_size=x.shape[1:], n_latent=n_latent)
  D = Adv_D_Experimental(n_classes=1, data_size=x.shape[1:])

  # print some infos
  print("G Net: ", G)
  print("D Net: ", D)

  # go through all modules
  for module in D.children():
    print("module cls: ", module.__class__.__name__)
    print("module: ", module)

  # test nets
  o_d = D(x)
  o_g = G(torch.randn(2, n_latent))

  # output
  print("d: ", o_d.shape)
  print("g: ", o_g.shape)
