"""
convolutional neural network architectures
partly adapted from pytorch tutorial

References:
[Sainath 2015] - Tara N. Sainath and Carolina Parada. Convolutional neural networks for small-footprint key-word spotting. InINTERSPEECH, 2015
"""

import numpy as np

import torch
import torch.nn as nn

from classifier_nets import ClassifierNetFc1, ClassifierNetFc3


class ConvBasics():
  """
  Convolutional Networks base class with useful functions
  """

  def __init__(self, n_classes, data_size):

    # arguments
    self.n_classes = n_classes
    self.data_size = data_size

    # extract input size [channel x features x frames]
    self.n_channels, self.n_features, self.n_frames = self.data_size

    # bias
    self.use_conv_bias = True


  def get_conv_layer_dimensions(self, input_dim, kernel_sizes, strides, padding):
    """
    get convolutional layer dimensions upon kernel sizes, strides and padding
    input_dim; (height, width) or (n_features, n_frames) for mfcc or (n_time_steps,) for raw input
    """

    # layer dimensions
    conv_layer_dim = []

    # first dimension
    conv_layer_dim.append(input_dim)

    # for all convolutional layers
    for i, (k, s, p) in enumerate(zip(kernel_sizes, strides, padding)):

      # init layer dim
      layer_dim = None

      # for all dimensions
      for d, (k_d, s_d, p_d) in enumerate(zip(k, s, p)):

        # dimension
        dim = int((conv_layer_dim[i][d] + 2 * p_d - k_d) / s_d + 1)

        # new dim
        layer_dim = layer_dim + (dim,) if layer_dim is not None else (dim,)

      # append to dimensions
      conv_layer_dim.append(layer_dim)

    return conv_layer_dim


  def calc_amount_of_params(self):
    """
    calculate amount of parameters
    """

    # convolution parameters
    n_params = {'conv{}'.format(i): np.prod(f) * np.prod(k) + (f[1] if self.use_conv_bias else 0) for i, (f, k) in enumerate(zip(self.n_feature_maps, self.kernel_sizes))}

    # fully connected parameters
    n_params.update({'fc{}'.format(i): m * n + n for i, (m, n) in enumerate(self.fc_layer_dims)})

    return n_params


  def count_params(self):
    """
    count all parameters
    """
    return [p.numel() for p in self.parameters() if p.requires_grad]


  def calc_amount_of_operations(self):
    """
    calculate amount of operations
    """

    # convolution operations
    n_ops = {'conv{}'.format(i): np.prod(f) * 2 * np.prod(o) * np.prod(k) + np.prod(f) * np.prod(o) for i, (o, f, k) in enumerate(zip(self.conv_layer_dim[1:], self.n_feature_maps, self.kernel_sizes))}

    # fully connected operations
    n_ops.update({'fc{}'.format(i): 2 * m * n + n for i, (m, n) in enumerate(self.fc_layer_dims)})

    return n_ops


  def transfer_params(self, model):
    """
    transfer parameters from other models
    """

    with torch.no_grad():

      # go through all parameters
      for param_name in model.state_dict():

        # encoders
        if param_name == 'conv_layer0.weight': self.state_dict()['conv_layer0.weight'][:] = model.state_dict()[param_name]
        elif param_name == 'conv_layer1.weight': self.state_dict()['conv_layer1.weight'][:] = model.state_dict()[param_name]

        # decoders
        elif param_name == 'deconv_layer0.weight': self.state_dict()['conv_layer1.weight'][:] = model.state_dict()[param_name]
        elif param_name == 'deconv_layer1.weight': self.state_dict()['conv_layer0.weight'][:] = model.state_dict()[param_name]


  def transfer_params_label_models(self, models, n=8):
    """
    transfer of parameters for adv-label models with n feature maps each
    """

    with torch.no_grad():

      # regard every conv encoder
      for i, model in enumerate(models):

        # go through all parameters
        for param_name in model.state_dict():

          # encoders
          if param_name == 'conv_layer0.weight': self.state_dict()[param_name][i*n:(i+1)*n] = model.state_dict()[param_name]
          elif param_name == 'conv_layer1.weight': self.state_dict()[param_name][:, i*n:(i+1)*n] = model.state_dict()[param_name]

          # decoders
          elif param_name == 'deconv_layer0.weight': self.state_dict()[param_name][:, i*n:(i+1)*n] = model.state_dict()[param_name]
          elif param_name == 'deconv_layer1.weight': self.state_dict()[param_name][i*n:(i+1)*n] = model.state_dict()[param_name]



class ConvNetTrad(nn.Module, ConvBasics):
  """
  Traditional CNN adapted from [Sainath 2015] - cnn-trad-fpool3
  """

  def __init__(self, n_classes, data_size):

    # parent init
    super().__init__()
    ConvBasics.__init__(self, n_classes, data_size)

    # original settings for 39x32
    #self.kernel_sizes = [(8, 20), (4, 1), (4, 10)]
    #self.strides = [(1, 1), (4, 1), (1, 1)]

    # params for 13x32
    self.n_feature_maps = [(self.n_channels, 64), (0, 0), (64, 64)]
    self.kernel_sizes = [(4, 20), (2, 4), (2, 4)]
    self.strides = [(1, 1), (2, 4), (1, 1)]
    self.padding = [(0, 0), (0, 0), (0, 0)]

    # get layer dimensions
    self.conv_layer_dim = self.get_conv_layer_dimensions(input_dim=(self.n_features, self.n_frames), kernel_sizes=self.kernel_sizes, strides=self.strides, padding=self.padding)
    self.conv_in_dim = self.data_size
    self.conv_out_dim = ((self.n_feature_maps[-1][1],) + self.conv_layer_dim[-1])

    # 1. conv layer
    self.conv1 = nn.Conv2d(self.n_feature_maps[0][0], self.n_feature_maps[0][1], kernel_size=self.kernel_sizes[0], stride=self.strides[0])

    # max pool layer
    self.pool = nn.MaxPool2d(kernel_size=self.kernel_sizes[1], stride=self.strides[1])

    # 2. conv layer
    self.conv2 = nn.Conv2d(self.n_feature_maps[2][0], self.n_feature_maps[2][1], kernel_size=self.kernel_sizes[2], stride=self.strides[2])

    # fc dims
    self.fc_layer_dims = [(np.prod(self.conv_out_dim), 32), (32, 128), (128, self.n_classes)]

    # fully connected layers
    self.fc1 = nn.Linear(np.prod(self.conv_out_dim), 32)
    self.fc2 = nn.Linear(32, 128)
    self.fc3 = nn.Linear(128, self.n_classes)

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
    x = torch.relu(self.conv1(x))

    # max pooling layer [1 x 64 x 8 x 13]
    x = self.pool(x)

    # 2. conv layer [1 x 64 x 5 x 4]
    x = torch.relu(self.conv2(x))

    # flatten output from 2. conv layer [1 x 1280]
    x = x.view(-1, np.product(x.shape[1:]))

    # 1. fully connected layers [1 x 32]
    x = self.fc1(x)
    x = self.dropout_layer1(x)

    # 2. fully connected layers [1 x 128]
    x = torch.relu(self.fc2(x))
    x = self.dropout_layer2(x)

    # Softmax layer [1 x n_classes]
    x = self.softmax(self.fc3(x))

    return x



class ConvNetFstride4(nn.Module, ConvBasics):
  """
  CNN architecture with limited multipliers adapted from [Sainath 2015] - cnn-one-fstride4
  """

  def __init__(self, n_classes, data_size):

    # parent init
    super().__init__()
    ConvBasics.__init__(self, n_classes, data_size)

    # params
    self.n_feature_maps = [(self.n_channels, 54)]
    self.kernel_sizes = [(8, self.n_frames)]
    self.strides = [(4, 1)]
    self.padding = [(0, 0)]

    # get layer dimensions
    self.conv_layer_dim = self.get_conv_layer_dimensions(input_dim=(self.n_features, self.n_frames), kernel_sizes=self.kernel_sizes, strides=self.strides, padding=self.padding)
    self.conv_in_dim = self.data_size
    self.conv_out_dim = ((self.n_feature_maps[-1][1],) + self.conv_layer_dim[-1])

    # fc dims
    self.fc_layer_dims = [(np.prod(self.conv_out_dim), 32), (32, 128), (128, 128), (128, self.n_classes)]

    # conv layer
    self.conv = nn.Conv2d(self.n_feature_maps[0][0], self.n_feature_maps[0][1], kernel_size=self.kernel_sizes[0], stride=self.strides[0])

    # fully connected layers with affine transformations: y = Wx + b
    self.fc1 = nn.Linear(np.prod(self.conv_out_dim), 32)
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
    x = torch.relu(self.conv(x))

    # flatten output from conv layer [1 x 432]
    x = x.view(-1, np.product(x.shape[1:]))

    # 1. fully connected layers [1 x 32]
    x = self.fc1(x)
    x = self.dropout_layer1(x)

    # 2. fully connected layers [1 x 128]
    x = torch.relu(self.fc2(x))
    x = self.dropout_layer2(x)

    # 3. fully connected layers [1 x 128]
    x = torch.relu(self.fc3(x))
    x = self.dropout_layer2(x)

    # Softmax layer [1 x n_classes]
    x = self.softmax(self.fc4(x))

    return x



class ConvNetExperimental(nn.Module, ConvBasics):
  """
  CNN experimental
  """

  def __init__(self, n_classes, data_size):

    # parent init
    super().__init__()
    ConvBasics.__init__(self, n_classes, data_size)

    # conv params
    self.n_feature_maps = [(self.n_channels, 4), (4, 8), (8, 5)]
    self.kernel_sizes = [(self.n_features, 20), (1, 6), (1, 9)]
    self.strides = [(1, 1), (1, 3), (1, 1)]
    self.padding = [(0, 0), (0, 0), (0, 0)]

    # relu params (be carefull, last layer should be false)
    self.relu_active = [True, True, False]
    self.dropout_active = [False, True, False]

    # get layer dimensions
    self.conv_layer_dim = self.get_conv_layer_dimensions(input_dim=(self.n_features, self.n_frames), kernel_sizes=self.kernel_sizes, strides=self.strides, padding=self.padding)

    # conv layer
    self.conv_layers = torch.nn.ModuleList()
    for f, k, s in zip(self.n_feature_maps, self.kernel_sizes, self.strides): self.conv_layers.append(nn.Conv2d(f[0], f[1], kernel_size=k, stride=s))

    # dimensions
    self.conv_in_dim = self.data_size
    self.conv_out_dim = ((self.n_feature_maps[-1][1],) + self.conv_layer_dim[-1])

    # softmax layer
    self.softmax = nn.Softmax(dim=1)


  def forward(self, x):
    """
    forward pass
    """

    # convolutional layers
    for conv, r, d in zip(self.conv_layers, self.relu_active, self.dropout_active):
      x = conv(x)
      if r: x = torch.relu(x)
      if d: x = self.dropout_layer2(x)

    # flatten
    x = x.view(-1, np.product(x.shape[1:]))

    # Softmax layer
    x = self.softmax(x)

    return x


class ConvJim(nn.Module, ConvBasics):
  """
  CNN encoder with fc3
  """

  def __init__(self, n_classes, data_size):

    # parent init
    super().__init__()
    ConvBasics.__init__(self, n_classes, data_size)

    # bias
    self.use_conv_bias = False

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
    self.conv_layer0 = nn.Conv2d(self.n_feature_maps[0][0], self.n_feature_maps[0][1], kernel_size=self.kernel_sizes[0], stride=self.strides[0], bias=self.use_conv_bias)
    self.conv_layer1 = nn.Conv2d(self.n_feature_maps[1][0], self.n_feature_maps[1][1], kernel_size=self.kernel_sizes[1], stride=self.strides[1], bias=self.use_conv_bias)

    # classifier net
    self.classifier_net = ClassifierNetFc3(np.prod(self.conv_out_dim), n_classes)

    # fc layer dims
    self.fc_layer_dims = self.classifier_net.fc_layer_dims


  def forward(self, x):
    """
    forward pass
    """

    # conv layers
    x = torch.relu(self.conv_layer0(x))
    x = torch.relu(self.conv_layer1(x))

    # flatten
    x = x.view(-1, np.product(x.shape[1:]))

    # classifier net
    x = self.classifier_net(x)

    return x



if __name__ == '__main__':
  """
  main function
  """

  # generate random sample
  x = torch.randn((1, 1, 12, 50))

  # create net
  #model = ConvNetTrad(n_classes=12, data_size=x.shape[1:])
  #model = ConvNetFstride4(n_classes=12, data_size=x.shape[1:])
  model = ConvJim(n_classes=12, data_size=x.shape[1:])

  # test net
  o = model(x)

  # get params and ops
  n_params = model.calc_amount_of_params()
  n_ops = model.calc_amount_of_operations()


  # print some infos
  print("\nx: ", x.shape), print("model: ", model), print("o: ", o)

  # print amount of operations and number of params
  print("\ndim: {}".format(model.conv_layer_dim))
  print("conv out dim: {} flatten: {}".format(model.conv_out_dim, np.prod(model.conv_out_dim)))
  print("\nall params: ", model.count_params())
  print("\nparams: {}, sum: {:,}".format(n_params, np.sum(list(n_params.values()))))
  print("\noperations: {}, sum: {:,}".format(n_ops, np.sum(list(n_ops.values()))))