"""
hybrid networks
"""

import numpy as np
import torch
import torch.nn as nn

from classifier_nets import ClassifierNetFc3, ClassifierNetFc1
from adversarial_nets import AdvBasics


class HybJim(nn.Module, AdvBasics):
  """
  hybrid jim network
  """

  def __init__(self, n_classes, data_size, n_latent=100, n_feature_maps_l0=48):

    # parent init
    super().__init__()
    AdvBasics.__init__(self, n_classes, data_size)

    # arguments
    self.n_classes = n_classes
    self.data_size = data_size
    self.n_latent = n_latent
    self.n_feature_maps_l0 = n_feature_maps_l0

    # conv params
    self.n_feature_maps = [(self.n_channels, 48), (48, 8)]
    self.kernel_sizes = [(self.n_features, 20), (1, 5)]
    self.strides = [(1, 1), (1, 1)]
    self.padding = [(0, 0), (0, 0)]

    # get conv layer dimensions
    self.conv_layer_dim = self.get_conv_layer_dimensions((self.n_features, self.n_frames), self.kernel_sizes, self.strides, self.padding)
    self.conv_in_dim = self.data_size
    self.conv_out_dim = ((self.n_feature_maps[-1][1],) + self.conv_layer_dim[-1])

    # conv layer
    self.conv_layer0 = nn.Conv2d(self.n_feature_maps[0][0], self.n_feature_maps[0][1], kernel_size=self.kernel_sizes[0], stride=self.strides[0], bias=False)
    self.conv_layer1 = nn.Conv2d(self.n_feature_maps[1][0], self.n_feature_maps[1][1], kernel_size=self.kernel_sizes[1], stride=self.strides[1], bias=False)

    # classifier net
    #self.classifier_net = ClassifierNetFc3(np.prod(self.conv_out_dim), n_classes, dropout_enabled=False)
    self.classifier_net = ClassifierNetFc1(np.prod(self.conv_out_dim), n_classes)

    # discriminator net
    self.discriminator_net = nn.Linear(np.prod(self.conv_out_dim), 1)

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

    # discriminator net
    y = torch.sigmoid(self.discriminator_net(x))

    # classifier net
    x = self.classifier_net(x)

    return x, y



if __name__ == '__main__':
  """
  main function
  """

  # generate random sample
  x = torch.randn((1, 1, 13, 50))

  # create net
  net = HybJim(n_classes=5, data_size=x.shape[1:])

  # test net
  o_x, o_y = net(x)

  # print some infos
  print("\nx: ", x.shape), print("Net: ", net), print("o: ", o_x), print("o: ", o_y)