"""
Machine Learning file for training and evaluating the model
"""

import numpy as np
import matplotlib.pyplot as plt

import torch

from skimage.util.shape import view_as_windows

# my stuff
from common import create_folder
from plots import plot_mfcc_only
from conv_nets import ConvNetTrad


def create_batches(x_data, y_data, index, f, batch_size):
  """
  create batches for training N x [b x m x f]
  x: [n x m x l]
  y: [n]
  N: Amount of batches
  b: batch size
  m: feature size
  f: frame length
  """

  # get shape of things
  n, m, l = x_data.shape

  # randomize data
  indices = np.random.permutation(x_data.shape[0])
  x_data = np.take(x_data, indices, axis=0)
  y_data = np.take(y_data, indices, axis=0)

  # x: [n x m x f]
  x = np.empty(shape=(0, 39, f), dtype=x_data.dtype)
  y = np.empty(shape=(0), dtype=y_data.dtype)

  i = 0
  # run through all samples
  for x_n, y_n in zip(x_data, y_data):

    # windowed [r x m x f]
    x_win = np.squeeze(view_as_windows(x_n, (m, f), step=1))

    # window length
    l_win = x_win.shape[0]

    # append y
    y = np.append(y, [y_n] * l_win)

    # stack windowed [n+r x m x f]
    x = np.vstack((x, x_win))

    #for i in range(r):
    #  plot_mfcc_only(x[i], fs, hop, plot_path, name=index[0] + str(i))

    # TODO: remove debug line later
    i += 1
    if i > 10:
      break

  # randomize examples
  indices = np.random.permutation(x.shape[0])
  x = np.take(x, indices, axis=0)
  y = np.take(y, indices, axis=0)


  # create batches

  print("x: ", x.shape)
  print("y: ", y.shape)

  return x, y


def one_hot_label(y, classes, to_torch=False):
  """
  create one hot encoded vector e.g.:
  classes = ['up', 'down']
  y = 'up'
  return [1, 0]
  """

  # create one hot vector
  hot = np.array([c == y for c in classes]).astype(int)

  # transfer to torch
  if to_torch:
    hot = torch.from_numpy(hot)

  return hot


def get_index_of_class(y, classes, to_torch=False):
  """
  return index of class
  """

  # get index
  y_idx = np.where(np.array(classes) == y)[0]

  # transfer to torch
  if to_torch:
    y_idx = torch.from_numpy(y_idx)

  return y_idx



def train_nn(x_batches, y_batches, nn_architecture, classes, num_epochs=2, lr=1e-3):
  """
  train the neural network
  """

  # select network architecture
  if nn_architecture == 'conv-trad':

    # traditional conv-net
    net = ConvNetTrad()

  else:

    # traditional conv-net
    net = ConvNetTrad()


  # MSE Loss
  criterion = torch.nn.CrossEntropyLoss()

  # create optimizer
  optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.5)

  print("\n--Training starts:")

  # epochs
  for epoch in range(num_epochs):

    # cumulated loss
    cum_loss = 0.0

    # TODO: do this with loader function from pytorch
    # fetch data samples
    for i, (x, y) in enumerate(zip(x_batches, y_batches)):

      # zero parameter gradients
      optimizer.zero_grad()

      # prepare x
      x = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(x.astype(np.float32)), 0), 0)

      # forward pass
      o = net(x)

      # get index of class
      y = get_index_of_class(y, classes, to_torch=True)

      # loss
      loss = criterion(o, y)

      # backward
      loss.backward()

      # optimizer step - update params
      optimizer.step()

      # loss update
      cum_loss += loss.item()

      # print every 2000 mini-batches
      if i % 10 == 9:

        # print info
        print('epoch: {}, mini-batch: {}, loss: [{:.5f}]'.format(epoch + 1, i + 1, cum_loss / 10))
        
        # zero cum loss
        cum_loss = 0.0
    
  print('Training finished')

  # save parameters of network
  #torch.save(net.state_dict(), param_path)


if __name__ == '__main__':
  """
  ML - Machine Learning file
  """

  # path to train, test and eval set
  mfcc_data_files = ['./ignore/train/mfcc_data_train_n-100_c-5.npz', './ignore/test/mfcc_data_test_n-100_c-5.npz', './ignore/eval/mfcc_data_eval_n-100_c-5.npz']

  # plot path
  plot_path = './ignore/plots/ml/'

  # model path
  model_path = "./ignore/models"

  # create folder
  create_folder([plot_path, model_path])

  # load files
  data = [np.load(file, allow_pickle=True) for file in mfcc_data_files]

  # training data
  train_data = data[0]

  print("train_data: ", train_data.files)

  # extract data from file
  x_data, y_data, index, info, params = train_data['x'], train_data['y'], train_data['index'], train_data['info'], train_data['params']

  # shape of things
  n, m, l = x_data.shape

  # get some params
  fs, hop = params[()]['fs'], params[()]['hop']

  # get classes
  classes = np.unique(y_data)
  print("classes: ", classes)

  #print("x_data: ", x_data.shape)
  #print("index: ", index)

  # frames for input
  f = 32

  # -- names:
  # n: samples
  # l: length of frames 
  # m: features per frame
  # r: stride of frames, 
  # f: frame length for input into NN
  #
  # -- shapes:
  # x:  [n x m x l]
  # xr: [n x m x r x f]


  # --
  # prepare data for training

  # plot mfcc only
  #plot_mfcc_only(x_data[0], fs, hop, plot_path, name=index[0])

  # batch size
  batch_size = 512

  # create batches
  x_batches, y_batches = create_batches(x_data, y_data, index, f, batch_size)

  print("x_batches: ", x_batches.shape)
  print("y_batches: ", y_batches.shape)


  # --
  # actual training

  # num epochs
  num_epochs = 2
  num_classes = len(classes)

  # nn architecture
  nn_architectures = ['conv-trad']

  # select architecture
  nn_architecture = nn_architectures[0]

  # train
  train_nn(x_batches, y_batches, nn_architecture, classes, num_epochs=num_epochs)






