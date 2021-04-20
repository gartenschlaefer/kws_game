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

    #[(self.n_channels, 8), (8, 4), (4, 5)]

    for i, (k, s) in enumerate(zip(self.kernel_sizes, self.strides)):
      self.conv_layer_dim.append((int((self.conv_layer_dim[i][0] - k[0]) / s[0] + 1), int((self.conv_layer_dim[i][1] - k[1]) / s[1] + 1)))



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


class ConvNetExperimental2(nn.Module, ConvBasics):
  """
  Convolutional net for experiments
  """

  def __init__(self, n_classes, data_size):

    # parent init
    super().__init__()
    ConvBasics.__init__(self, n_classes, data_size)

    # params
    # self.n_feature_maps = (1)
    # self.kernel_sizes = [(self.n_features, 20)]
    # self.strides = [(1, 1)]

    # # conv params
    # self.n_feature_maps = [(self.n_channels, 5), (5, 4), (4, 5)]
    # self.kernel_sizes = [(self.n_features, 20), (1, 5), (1, 6)]
    # self.strides = [(1, 1), (1, 5), (1, 1)]

    # conv params
    self.n_feature_maps = [(self.n_channels, 4), (4, 8), (8, 5)]
    self.kernel_sizes = [(self.n_features, 20), (1, 6), (1, 9)]
    self.strides = [(1, 1), (1, 3), (1, 1)]

    # relu params (be carefull, last layer should be false)
    self.relu_active = [True, True, False]
    self.dropout_active = [False, True, False]

    # self.n_feature_maps = [(1, 4), (4, 2)]
    # self.kernel_sizes = [(self.n_features, 20), (1, 5)]
    # self.strides = [(1, 1), (1, 5)]

    # self.relu_active = [True, False]
    # self.dropout_active = [False, False]

    # get layer dimensions
    self.get_conv_layer_dimensions()
    print("conv: ", self.conv_layer_dim)


    # conv layer
    self.conv_layers = torch.nn.ModuleList()
    for f, k, s in zip(self.n_feature_maps, self.kernel_sizes, self.strides):
      self.conv_layers.append(nn.Conv2d(f[0], f[1], kernel_size=k, stride=s))

    # dimensions
    self.conv_in_dim = self.data_size
    self.conv_out_dim = ((self.n_feature_maps[-1][1],) + self.conv_layer_dim[-1])

    # conv layer
    #self.conv1 = nn.Conv2d(self.n_channels, self.n_feature_maps[0], kernel_size=self.kernel_sizes[0], stride=self.strides[0])
    #self.bn1 = nn.BatchNorm2d(self.n_feature_maps[0])

    #self.conv2 = nn.Conv2d(self.n_feature_maps[0], self.n_feature_maps[1], kernel_size=self.kernel_sizes[1], stride=self.strides[1])
    #self.bn2 = nn.BatchNorm2d(self.n_classes)

    # fully connected layers with affine transformations: y = Wx + b
    #self.fc1 = nn.Linear(np.prod(self.conv_out_dim), self.n_classes)

    # two fully connected
    #self.fc1 = nn.Linear(np.prod(self.conv_layer_dim[-1]) * self.n_feature_maps[-1], 32)
    #self.fc2 = nn.Linear(32, self.n_classes)

    # dropout layer
    #self.dropout_layer1 = nn.Dropout(p=0.5)
    self.dropout_layer2 = nn.Dropout(p=0.2)

    # softmax layer
    self.softmax = nn.Softmax(dim=1)


  def forward(self, x):
    """
    forward pass
    """

    # convolutional layers
    for conv, r, d in zip(self.conv_layers, self.relu_active, self.dropout_active):
      x = conv(x)
      if r: x = F.relu(x)
      if d: x = self.dropout_layer2(x)

    #print("x: ", x.shape), adfasdf
    # flatten output from conv layer
    x = x.view(-1, np.product(x.shape[1:]))

    # 1. fully connected layers [1 x 32]
    #x = self.fc1(x)
    #x = F.relu(x)
    #x = self.dropout_layer1(x)

    # 2. fully connected layer
    #x = self.fc2(x)

    # Softmax layer
    x = self.softmax(x)
    #x = torch.squeeze(x, axis=-1)

    return x


class ConvNetExperimental(nn.Module, ConvBasics):
  """
  Convolutional net for experiments
  """

  def __init__(self, n_classes, data_size):

    # parent init
    super().__init__()
    ConvBasics.__init__(self, n_classes, data_size)

    # encoder model
    self.conv_encoder = ConvEncoder(self.n_classes, self.data_size)

    # fully connected layers
    self.fc1 = nn.Linear(np.prod(self.conv_encoder.conv_out_dim), self.n_classes)

    # softmax layer
    self.softmax = nn.Softmax(dim=1)


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

    # Softmax layer
    x = self.softmax(x)

    return x


class ConvEncoderDecoderParams(ConvBasics):
  """
  parameters for experimental
  """

  #def __init__(self, n_classes, data_size, is_collection_net=False):
  def __init__(self, n_classes, data_size, net_class='label-encoder'):

    # parent init
    super().__init__(n_classes, data_size)

    # arguments
    self.net_class = net_class

    # feature maps regarding
    if self.net_class == 'label-encoder': self.n_feature_maps = [(self.n_channels, 8), (8, 8)]
    elif self.net_class == 'label-collect-encoder': self.n_feature_maps = [(self.n_channels, 8 * self.n_classes), (8 * self.n_classes, 8)]
    elif self.net_class == 'lim-encoder': self.n_feature_maps = [(self.n_channels, 8 * 5), (8 * 5, 8)]

    # conv params
    self.kernel_sizes = [(self.n_features, 20), (1, 5)]
    self.strides = [(1, 1), (1, 1)]

    # relu params (be carefull, last layer should be false)
    self.relu_active = [True, False]

    # get layer dimensions
    self.get_conv_layer_dimensions()


  def transfer_conv_weights_label_coders(self, conv_coders):
    """
    transfer convolutional weights from encoder models to one concatenated model (use with care)
    """

    with torch.no_grad():

      # regard every conv encoder
      for i, conv_coder in enumerate(conv_coders):

        #print("stat: ", conv_coder.state_dict().keys())
        # go through all parameters
        for param_tensor in conv_coder.state_dict():

          # encoder fist layer
          if param_tensor == 'conv_layers.0.weight':
            self.state_dict()[param_tensor][i*8:(i+1)*8] = conv_coder.state_dict()[param_tensor]

          # encoder second layer
          elif param_tensor == 'conv_layers.1.weight':
            self.state_dict()[param_tensor][:, i*8:(i+1)*8] = conv_coder.state_dict()[param_tensor]

          # decoder first layer
          elif param_tensor == 'deconv_layers.0.weight':
            self.state_dict()[param_tensor][:, i*8:(i+1)*8] = conv_coder.state_dict()[param_tensor]

          # decoder second layer
          elif param_tensor == 'deconv_layers.1.weight':
            self.state_dict()[param_tensor][i*8:(i+1)*8] = conv_coder.state_dict()[param_tensor]


  def transfer_decoder_weights(self, conv_decoder):
    """
    transfer decoder weights (experimental) 
    """

    with torch.no_grad():

      # go through all parameters
      for param_tensor in conv_decoder.state_dict():

        # first layer
        if param_tensor == 'deconv_layers.0.weight':
          self.state_dict()['conv_layers.1.weight'][:] = conv_decoder.state_dict()[param_tensor]

        # second layer
        elif param_tensor == 'deconv_layers.1.weight':
          #print("yoa")
          self.state_dict()['conv_layers.0.weight'][:] = conv_decoder.state_dict()[param_tensor]



class ConvEncoder(nn.Module, ConvEncoderDecoderParams):
  """
  Convolutional encoder for discriminator
  """

  def __init__(self, n_classes, data_size, net_class='label-encoder'):

    # parent init
    super().__init__()
    ConvEncoderDecoderParams.__init__(self, n_classes, data_size, net_class)

    # conv layer
    self.conv_layers = torch.nn.ModuleList()
    for f, k, s in zip(self.n_feature_maps, self.kernel_sizes, self.strides):
      #self.conv_layers.append(nn.Conv2d(f[0], f[1], kernel_size=k, stride=s))
      self.conv_layers.append(nn.Conv2d(f[0], f[1], kernel_size=k, stride=s, bias=False))

    # dimensions
    self.conv_in_dim = self.data_size
    self.conv_out_dim = ((self.n_feature_maps[-1][1],) + self.conv_layer_dim[-1])

    # dropout layer
    #self.dropout_layer1 = nn.Dropout(p=0.5)


  def forward(self, x):
    """
    forward pass
    """

    # convolutional layers
    for conv, r in zip(self.conv_layers, self.relu_active):
      x = conv(x)
      if r: x = F.relu(x)

    # last relu
    x = F.relu(x)

    # dropout
    #x = self.dropout_layer1(x)

    return x



class ConvDecoder(nn.Module, ConvEncoderDecoderParams):
  """
  Convolutional decoder for adversarial nets Generator
  """

  def __init__(self, n_classes, data_size, n_latent=100, net_class='label-encoder'):

    # parent init
    super().__init__()
    ConvEncoderDecoderParams.__init__(self, n_classes, data_size, net_class)

    # adapt params if necessary
    #self.n_feature_maps = [(self.n_channels, 12), (12, 8)]

    # arguments
    self.n_latent = n_latent

    # deconv layers module list
    self.deconv_layers = torch.nn.ModuleList()

    # create deconv layers
    for f, k, s in zip(reversed(self.n_feature_maps), reversed(self.kernel_sizes), reversed(self.strides)):
      self.deconv_layers.append(nn.ConvTranspose2d(in_channels=f[1], out_channels=f[0], kernel_size=k, stride=s, bias=False))

    # dimensions
    self.conv_in_dim = ((self.n_feature_maps[-1][1],) + self.conv_layer_dim[-1])
    self.conv_out_dim = self.data_size


  def forward(self, x):
    """
    forward pass
    """

    # deconvolutional layers
    for deconv, r in zip(self.deconv_layers, self.relu_active):
      x = deconv(x)
      if r: x = F.relu(x)

    return x



class ClassifierNet(nn.Module):
  """
  Classifier Network
  """

  def __init__(self, input_dim, output_dim):

    # parent init
    super().__init__()

    # # fully connected layers
    # self.fc1 = nn.Linear(input_dim, 64)
    # self.fc2 = nn.Linear(64, 32)
    # self.fc3 = nn.Linear(32, output_dim)

    self.fc1 = nn.Linear(input_dim, output_dim)

    # dropout layer
    #self.dropout_layer1 = nn.Dropout(p=0.2)
    self.dropout_layer2 = nn.Dropout(p=0.5)

    # softmax layer
    self.softmax = nn.Softmax(dim=1)


  def forward(self, x):
    """
    forward pass
    """

    # # 1. fully connected layer
    # x = self.fc1(x)
    # x = F.relu(x)

    # # 2. fully connected layer
    # x = self.fc2(x)
    # x = F.relu(x)
    # #x = self.dropout_layer1(x)
    # x = self.dropout_layer2(x)

    # # 3. fully connected layer
    # x = self.fc3(x)

    x = self.fc1(x)

    # Softmax layer
    x = self.softmax(x)

    return x



class ClassifierNetFc1(nn.Module):
  """
  Classifier Network
  """

  def __init__(self, input_dim, output_dim):

    # parent init
    super().__init__()

    # structure
    self.fc1, self.softmax = nn.Linear(input_dim, output_dim), nn.Softmax(dim=1)


  def forward(self, x):
    """
    forward pass
    """
    return self.softmax(self.fc1(x))



class ClassifierNetFc3(nn.Module):
  """
  Classifier Network
  """

  def __init__(self, input_dim, output_dim):

    # parent init
    super().__init__()

    # structure
    self.fc1, self.fc2, self.fc3, self.dropout_layer, self.softmax = nn.Linear(input_dim, 64), nn.Linear(64, 32), nn.Linear(32, output_dim), nn.Dropout(p=0.5), nn.Softmax(dim=1)


  def forward(self, x):
    """
    forward pass
    """
    return self.softmax(self.fc3(self.dropout_layer(F.relu(self.fc2(F.relu(self.fc1(x)))))))



class ConvStackedEncodersNet(nn.Module, ConvBasics):
  """
  Collected encoder networks with consecutive classifier Network
  """

  def __init__(self, n_classes, data_size, encoder_model):

    # parent init
    super().__init__()
    ConvBasics.__init__(self, n_classes, data_size)

    # arguments
    self.encoder_model = encoder_model

    # get flatten output dim
    self.flatten_encoder_dim = np.sum([np.prod(encoder_model.conv_out_dim) for encoder_model in self.encoder_model])

    # classifier net
    self.classifier_net = ClassifierNet(self.flatten_encoder_dim, n_classes)


  def forward(self, x):
    """
    forward pass
    """

    # encoder models models in parallel
    x = torch.cat([m(x) for m in self.encoder_model], dim=1)

    # flatten output from conv layer [1 x 432]
    x = x.view(-1, np.product(x.shape[1:]))

    # classifier net
    x = self.classifier_net(x)

    return x



class ConvEncoderClassifierNet(nn.Module, ConvBasics):
  """
  Collected encoder networks with consecutive classifier Network
  """

  def __init__(self, n_classes, data_size, net_class='label-collect-encoder', fc_layer_type='fc3'):

    # parent init
    super().__init__()
    ConvBasics.__init__(self, n_classes, data_size)

    self.conv_encoder = ConvEncoder(n_classes, data_size, net_class=net_class)

    # dimensions
    self.conv_in_dim, self.conv_out_dim  = self.data_size, ((self.conv_encoder.n_feature_maps[-1][1],) + self.conv_encoder.conv_layer_dim[-1])

    # classifier net
    self.classifier_net = ClassifierNetFc3(np.prod(self.conv_out_dim), n_classes) if fc_layer_type == 'fc3' else ClassifierNetFc1(np.prod(self.conv_out_dim), n_classes)


  def forward(self, x):
    """
    forward pass
    """

    # encoder model
    x = self.conv_encoder(x)

    # flatten output from conv layer
    x = x.view(-1, np.product(x.shape[1:]))

    # classifier net
    x = self.classifier_net(x)

    return x
 


class ConvLatentClassifier(nn.Module, ConvBasics):
  """
  Collected encoder networks with consecutive classifier Network
  """

  def __init__(self, n_classes, data_size, net_class='lim-encoder', n_latent=100):

    # parent init
    super().__init__()
    ConvBasics.__init__(self, n_classes, data_size)

    # encoder model
    self.conv_encoder = ConvEncoder(n_classes, data_size, net_class=net_class)

    # dimensions
    self.conv_in_dim, self.conv_out_dim  = self.data_size, ((self.conv_encoder.n_feature_maps[-1][1],) + self.conv_encoder.conv_layer_dim[-1])

    # latent
    self.fc_latent = self.fc1 = nn.Linear(np.prod(self.conv_out_dim), n_latent)

    # classifier net
    self.classifier_net = ClassifierNet(n_latent, n_classes)


  def forward(self, x):
    """
    forward pass
    """

    # encoder model
    x = self.conv_encoder(x)

    # flatten output from conv layer
    x = x.view(-1, np.product(x.shape[1:]))

    # latent space
    x = self.fc_latent(x)

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
  print("\nx: ", x.shape), print("Net: ", net), print("o: ", o)