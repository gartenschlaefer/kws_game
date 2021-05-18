"""
mini convolutional net experiment
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image

import torch
import torch.nn as nn

import sys
from glob import glob

sys.path.append("../")

from batch_archive import BatchArchive
from net_handler import CnnHandler
from conv_nets import ConvBasics


class PixelCharactersBatchArchive(BatchArchive):
  """
  Batch Archiv for Pixel Characters
  """
  def __init__(self, batch_size=1, batch_size_eval=1, to_torch=True):

    # parent init
    super().__init__(batch_size, batch_size_eval)


class ConvPixelCharacters(nn.Module, ConvBasics):
  """
  Conv Net architecture
  """

  def __init__(self, n_classes, data_size):
    """
    define neural network architecture
    input: [batch x channels x m x f]
    m - features (MFCC)
    f - frames
    """

    # parent init
    super().__init__()
    ConvBasics.__init__(self, n_classes, data_size)

    # conv params
    self.n_feature_maps = [(4, 8)]
    self.kernel_sizes = [(5, 5)]
    self.strides = [(1, 1)]

    # relu params (be carefull, last layer should be false)
    self.relu_active = [True]

    # get layer dimensions
    self.get_conv_layer_dimensions()

    # conv layer
    self.conv_layers = torch.nn.ModuleList()
    for f, k, s in zip(self.n_feature_maps, self.kernel_sizes, self.strides):
      self.conv_layers.append(nn.Conv2d(f[0], f[1], kernel_size=k, stride=s, bias=False))

    # dimensions
    self.conv_in_dim = self.data_size
    self.conv_out_dim = ((self.n_feature_maps[-1][1],) + self.conv_layer_dim[-1])

    # fully connected layers with affine transformations: y = Wx + b
    self.fc1 = nn.Linear(np.prod(self.conv_out_dim), self.n_classes)

    # softmax layer
    self.softmax = nn.Softmax(dim=1)


  def forward(self, x):
    """
    forward pass
    """

    # convolutional layers
    for conv, r in zip(self.conv_layers, self.relu_active):
      x = conv(x)
      if r: x = torch.relu(x)

    # flatten output from conv layer
    x = x.view(-1, np.product(x.shape[1:]))

    # fc
    x = self.fc1(x)

    # softmax
    x = self.softmax(x)

    return x


if __name__ == '__main__':
  """
  main function
  """

  import yaml

  # yaml config file
  cfg = yaml.safe_load(open("../config.yaml"))

  # get all class directories except the ones starting with _
  class_dirs = glob('./ignore/data/' + '[!_]*/')

  # labels
  labels = []

  # data
  image_data = np.empty(shape=(0, 16, 16, 4), dtype=np.float32)
  misc_data = np.empty(shape=(0, 16, 16, 4), dtype=np.float32)

  label_data = []
  label_misc = []

  # run through all class directories
  for class_dir in class_dirs:

    print("class dir: ", class_dir)

    # extract label
    label = class_dir.split('/')[-2]

    # append to label list
    labels.append(label)

    # get all .wav files
    images = glob(class_dir + '*.png')

    for img in images:
      if label != 'misc':
        print("img: ", img)
        image_data = np.vstack((image_data, matplotlib.image.imread(img)[np.newaxis, :, :, :]))
        label_data.append(label)

      else:
        misc_data = np.vstack((misc_data, matplotlib.image.imread(img)[np.newaxis, :, :, :]))
        label_misc.append([0.5, 0.5])


  # create batch archiv
  batch_archive = PixelCharactersBatchArchive(batch_size=1, batch_size_eval=1, to_torch=True)
  batch_archive.create_class_dictionary(label_data)

  # [num_batches x batch_size x channel x 39 x 32]
  batch_archive.x_train = torch.unsqueeze(torch.from_numpy(image_data.astype(np.float32)).permute(0, 3, 1, 2), 0).contiguous()

  # get labels
  batch_archive.y_train = torch.unsqueeze(batch_archive.get_index_of_class(label_data, to_torch=True), 0)

  # validation set is train set
  batch_archive.x_val = batch_archive.x_train
  batch_archive.y_val = batch_archive.y_train

  batch_archive.data_size = (4, 16, 16)

  print("batch_archive: ", batch_archive.class_dict)
  print("batch_archive.x_train: ", batch_archive.x_train.shape)
  print("batch_archive.y_train: ", batch_archive.y_train)
  print("data size: ", batch_archive.data_size)

  # net handler
  net_handler = CnnHandler(nn_arch='mini', class_dict=batch_archive.class_dict, data_size=batch_archive.data_size, use_cpu=True)

  # init model
  net_handler.models = {'cnn':ConvPixelCharacters(n_classes=2, data_size=batch_archive.data_size)}
  net_handler.init_models()
  #net_handler.model.to(net_handler.device)
  #net_handler.model(batch_archive.x_train[0])


  train_params = cfg['ml']['train_params']
  train_params['num_epochs'] = 100
  train_params['lr'] = 0.0001

  # training
  net_handler.train_nn(cfg['ml']['train_params'], batch_archive=batch_archive)

  # classify samples
  c_img = torch.unsqueeze(batch_archive.x_train[0, 0], 0)
  #print("c_img1: ", c_img.shape)

  o = net_handler.models['cnn'](c_img)
  print("\nc_img1: ", o)



  x_misc = torch.unsqueeze(torch.from_numpy(misc_data.astype(np.float32)).permute(0, 3, 1, 2), 0).contiguous()
  y_misc = torch.tensor(label_misc)

  misc_img = x_misc[0]

  print("x_misc: ", x_misc.shape)
  print("y_misc: ", y_misc.shape)


  o = net_handler.models['cnn'](misc_img)
  print("misc image 1: ", o)



  # mse
  criterion = torch.nn.MSELoss()

  # optim
  optimizer = torch.optim.Adam(net_handler.models['cnn'].parameters(), lr=train_params['lr'])

  # epochs
  for epoch in range(train_params['num_epochs']):

    # TODO: do this with loader function from pytorch (maybe or not)
    # fetch data samples
    for i, (x, y) in enumerate(zip(x_misc.to(net_handler.device), y_misc.to(net_handler.device))):

      # zero parameter gradients
      optimizer.zero_grad()

      # forward pass o:[b x c]
      o = net_handler.models['cnn'](x)

      print("o: ", o)
      # loss
      loss = criterion(o, y)
      print("loss: ", loss)

      # backward
      loss.backward()

      # optimizer step - update params
      optimizer.step()

    # eval
    #net_handler.eval_nn('val', batch_archive)



  o = net_handler.models['cnn'](c_img)
  print("\nc_img2: ", o)

  o = net_handler.models['cnn'](misc_img)
  print("misc image 2: ", o)


  #plt.imshow(batch_archive.x_train[0, 5].permute(1, 2, 0))
  #plt.imshow(misc_img[0].permute(1, 2, 0))
  plt.show()