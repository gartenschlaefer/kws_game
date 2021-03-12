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


class ConvBasics():
  """
  Convolutional Networks Basics with useful functions
  """

  def __init__(self, n_classes, data_size):

    # arguments
    self.n_classes = n_classes
    self.data_size = data_size

    # extract input size [channel x features x frames]
    self.n_channels, self.n_features, self.n_frames = self.data_size


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


class ConvNetTrad(nn.Module, ConvBasics):
  """
  Traditional Conv Net architecture 
  presented in [Sainath 2015] - cnn-trad-fpool3

  input: [batch x channels x m x f]
  m - features (MFCC)
  f - frames
  """

  def __init__(self, n_classes, data_size):

    # parent init
    super().__init__()
    ConvBasics.__init__(self, n_classes, data_size)

    # params
    self.n_feature_maps = [64, 64]

    # for 39x32
    #self.kernel_sizes = [(8, 20), (4, 1), (4, 10)]
    #self.strides = [(1, 1), (4, 1), (1, 1)]

    # for 13x32
    self.kernel_sizes = [(4, 20), (2, 4), (2, 4)]
    self.strides = [(1, 1), (2, 4), (1, 1)]

    # get layer dimensions
    self.get_conv_layer_dimensions()

    # 1. conv layer
    self.conv1 = nn.Conv2d(self.n_channels, self.n_feature_maps[0], kernel_size=self.kernel_sizes[0], stride=self.strides[0])

    # max pool layer
    self.pool = nn.MaxPool2d(kernel_size=self.kernel_sizes[1], stride=self.strides[1])

    # 2. conv layer
    self.conv2 = nn.Conv2d(64, 64, kernel_size=self.kernel_sizes[2], stride=self.strides[2])

    # fully connected layers with affine transformations: y = Wx + b
    self.fc1 = nn.Linear(np.prod(self.conv_layer_dim[-1]) * self.n_feature_maps[-1], 32)
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



class ConvNetFstride4(nn.Module, ConvBasics):
  """
  Conv Net architecture with limited multipliers 
  presented in [Sainath 2015] - cnn-one-fstride4

  input: [batch x channels x m x f]
  m - features (MFCC)
  f - frames
  """

  def __init__(self, n_classes, data_size):

    # parent init
    super().__init__()
    ConvBasics.__init__(self, n_classes, data_size)

    # params
    self.n_feature_maps = [54]
    self.kernel_sizes = [(8, self.n_frames)]
    self.strides = [(4, 1)]

    # get layer dimensions
    self.get_conv_layer_dimensions()

    # conv layer
    self.conv = nn.Conv2d(self.n_channels, self.n_feature_maps[0], kernel_size=self.kernel_sizes[0], stride=self.strides[0])

    # fully connected layers with affine transformations: y = Wx + b
    self.fc1 = nn.Linear(np.prod(self.conv_layer_dim[-1]) * self.n_feature_maps[-1], 32)
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



class ConvNetExperimental1(nn.Module, ConvBasics):
  """
  Convolutional Net for experiments
  """

  def __init__(self, n_classes, data_size):

    # parent init
    super().__init__()
    ConvBasics.__init__(self, n_classes, data_size)

    # params
    self.n_feature_maps = [16]
    self.kernel_sizes = [(self.n_features, 20)]
    self.strides = [(1, 1)]

    # get layer dimensions
    self.get_conv_layer_dimensions()

    # conv layer
    self.conv = nn.Conv2d(self.n_channels, self.n_feature_maps[0], kernel_size=self.kernel_sizes[0], stride=self.strides[0])

    # fully connected layers with affine transformations: y = Wx + b
    self.fc1 = nn.Linear(np.prod(self.conv_layer_dim[-1]) * self.n_feature_maps[-1], 32)
    self.fc2 = nn.Linear(32, self.n_classes)

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

    # Softmax layer [1 x n_classes]
    x = self.softmax(self.fc2(x))

    return x


  def get_weights(self):
    """
    get weights of model
    """
    return {'conv1': self.conv.weight.detach().cpu()}



class ConvNetExperimental(nn.Module, ConvBasics):
  """
  Convolutional net for experiments
  """

  def __init__(self, n_classes, data_size):

    # parent init
    super().__init__()
    ConvBasics.__init__(self, n_classes, data_size)

    # params
    # self.n_feature_maps = [16]
    # self.kernel_sizes = [(self.n_features, 20)]
    # self.strides = [(1, 10)]

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
    #self.fc1 = nn.Linear(np.prod(self.conv_layer_dim[-1]) * self.n_feature_maps[-1], 32)
    self.fc1 = nn.Linear(np.prod(self.conv_layer_dim[-1]) * self.n_feature_maps[-1], self.n_classes)

    self.fc2 = nn.Linear(32, self.n_classes)

    # dropout layer
    self.dropout_layer1 = nn.Dropout(p=0.5)
    self.dropout_layer2 = nn.Dropout(p=0.2)

    # softmax layer
    self.softmax = nn.Softmax(dim=1)


  def forward(self, x):
    """
    forward pass
    """

    # 1. conv layer
    x = self.conv1(x)
    #x = self.bn1(x)
    x = F.relu(x)
    #x = self.dropout_layer2(x)
    #print("x: ", x.shape)

    # 2. conv layer
    x = self.conv2(x)
    #x = self.bn2(x)
    x = F.relu(x)
    x = self.dropout_layer1(x)

    # flatten output from conv layer
    x = x.view(-1, np.product(x.shape[1:]))

    # 1. fully connected layers [1 x 32]
    x = self.fc1(x)
    #x = F.relu(x)
    #x = self.dropout_layer1(x)

    # 2. fully connected layer
    #x = self.fc2(x)

    # Softmax layer
    x = self.softmax(x)
    #x = torch.squeeze(x, axis=-1)
    #x = torch.squeeze(x, axis=-1)

    return x


  def get_weights(self):
    """
    get weights of model
    """
    return {'conv1': self.conv1.weight.detach().cpu()}
    


class ConvEncoderDecoderParams(ConvBasics):
  """
  parameters for experimental
  """

  def __init__(self, n_classes, data_size):

    # parent init
    super().__init__(n_classes, data_size)

    # params (reversed order)
    self.n_feature_maps = [8, 4]
    self.kernel_sizes = [(self.n_features, 20), (1, 5)]
    self.strides = [(1, 1), (1, 5)]

    # get layer dimensions
    self.get_conv_layer_dimensions()



class ConvEncoder(nn.Module, ConvEncoderDecoderParams):
  """
  Convolutional encoder
  """

  def __init__(self, n_classes, data_size):

    # parent init
    super().__init__()
    ConvEncoderDecoderParams.__init__(self, n_classes, data_size)

    # conv layer
    self.conv1 = nn.Conv2d(self.n_channels, self.n_feature_maps[0], kernel_size=self.kernel_sizes[0], stride=self.strides[0])
    self.conv2 = nn.Conv2d(self.n_feature_maps[0], self.n_feature_maps[1], kernel_size=self.kernel_sizes[1], stride=self.strides[1])

    # dropout layer
    self.dropout_layer1 = nn.Dropout(p=0.5)


  def forward(self, x):
    """
    forward pass
    """

    # 1. conv layer
    x = self.conv1(x)
    x = F.relu(x)

    # 2. conv layer
    x = self.conv2(x)
    x = F.relu(x)
    x = self.dropout_layer1(x)

    return x



class ConvDecoder(nn.Module, ConvEncoderDecoderParams):
  """
  Convolutional decoder for adversarial nets
  """

  def __init__(self, n_classes, data_size):

    # parent init
    super().__init__()
    ConvEncoderDecoderParams.__init__(self, n_classes, data_size)

    # conv layer
    self.deconv1 = nn.ConvTranspose2d(in_channels=self.n_feature_maps[1], out_channels=self.n_feature_maps[0], kernel_size=self.kernel_sizes[1], stride=self.strides[1])
    #self.bn1 = nn.BatchNorm2d(self.n_feature_maps[0])

    self.deconv2 = nn.ConvTranspose2d(in_channels=self.n_feature_maps[0], out_channels=1, kernel_size=self.kernel_sizes[0], stride=self.strides[0])
    #self.bn2 = nn.BatchNorm2d(self.n_classes)


  def forward(self, x):
    """
    forward pass
    """

    # 1. deconv layer
    x = self.deconv1(x)
    x = F.relu(x)

    # 2. deconv layer
    x = self.deconv2(x)

    return x



class CollectedConvEncoderNet(nn.Module):
  """
  Collected encoder networks
  """

  def __init__(self, encoder_models):

    # parent init
    super().__init__()

    # arguments
    self.encoder_models = encoder_models

    # get last dimension and feature map
    self.last_conv_layer_dims = [encoder_model.conv_layer_dim[-1] for encoder_model in self.encoder_models]
    self.last_feature_maps = [encoder_model.n_feature_maps[-1] for encoder_model in self.encoder_models]

    # calculate flatten output dimension
    self.flatten_output_dim = np.sum([np.prod(d + (f,)) for d, f in zip(self.last_conv_layer_dims, self.last_feature_maps)])

    # eval mode
    #self.encoder_models = [encoder_model.eval() for encoder_model in self.encoder_models]


  def forward(self, x):
    """
    forward pass
    """

    # models in parallel
    x = torch.cat([encoder_model(x) for encoder_model in self.encoder_models], dim=1)
    #for encoder_model in self.encoder_models:
    #  x = encoder_model(x)

    return x



class ClassifierNet(nn.Module):
  """
  Classifier Network
  """

  def __init__(self, input_dim, output_dim):

    # parent init
    super().__init__()

    # fully connected layers with affine transformations: y = Wx + b
    self.fc1 = nn.Linear(input_dim, 64)
    self.fc2 = nn.Linear(64, 32)
    self.fc3 = nn.Linear(32, output_dim)

    # dropout layer
    self.dropout_layer1 = nn.Dropout(p=0.2)
    #self.dropout_layer2 = nn.Dropout(p=0.5)

    # softmax layer
    self.softmax = nn.Softmax(dim=1)


  def forward(self, x):
    """
    forward pass
    """

    # 1. fully connected layers [1 x 32]
    x = self.fc1(x)
    x = F.relu(x)

    # 2. fully connected layers [1 x 128]
    x = self.fc2(x)
    x = F.relu(x)
    x = self.dropout_layer1(x)

    x = self.fc3(x)

    # Softmax layer [1 x n_classes]
    x = self.softmax(x)

    return x



class ConvEncoderClassifierNet(nn.Module, ConvBasics):
  """
  Collected encoder networks with consecutive classifier Network
  """

  def __init__(self, n_classes, data_size, encoder_models):

    # parent init
    super().__init__()
    ConvBasics.__init__(self, n_classes, data_size)

    # arguments
    self.encoder_models = encoder_models

    # conv encoder network
    self.conv_encoder_net = CollectedConvEncoderNet(encoder_models)

    # classifier net
    self.classifier_net = ClassifierNet(self.conv_encoder_net.flatten_output_dim, n_classes)


  def forward(self, x):
    """
    forward pass
    """

    # parallel conv encoder models
    x = self.conv_encoder_net(x)

    # flatten output from conv layer [1 x 432]
    x = x.view(-1, np.product(x.shape[1:]))

    # classifier net
    x = self.classifier_net(x)

    return x
 


if __name__ == '__main__':
  """
  main function
  """

  # generate random sample
  #x = torch.randn((1, 1, 39, 32))
  x = torch.randn((1, 1, 13, 50))

  # create net
  #net = ConvNetFstride4(n_classes=5, data_size=x.shape[1:])
  net = ConvNetTrad(n_classes=5, data_size=x.shape[1:])

  # test net
  o = net(x)

  # print some infos
  print("\nx: ", x.shape)
  print("Net: ", net)
  print("o: ", o)