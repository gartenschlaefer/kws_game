"""
Machine Learning file for training and evaluating the model
"""

import numpy as np
import matplotlib.pyplot as plt

import os
import torch

from skimage.util.shape import view_as_windows

# my stuff
from common import *
from plots import plot_mfcc_only
from conv_nets import ConvNetTrad

import logging
import time


def extract_to_batches(mfcc_data_files, f=32, batch_size=1, window_step=1):
  """
  extract data samples from file
  """

    # load files [0]: train, etc.
  data = [np.load(file, allow_pickle=True) for file in mfcc_data_files]

  # extract data
  train_data, test_data, eval_data = data[0], data[1], data[2]

  print("Container of train_data: ", train_data.files)

  # extract data from file
  index, info, params = train_data['index'], train_data['info'], train_data['params']

  # shape of things
  n, m, l = train_data['x'].shape

  # get some params
  fs, hop = params[()]['fs'], params[()]['hop']

  # get classes
  classes = np.unique(train_data['y'])
  print("classes: ", classes)

  # create batches
  x_train, y_train = create_batches(train_data['x'], train_data['y'], train_data['index'], f, batch_size, window_step)
  x_val, y_val = create_batches(eval_data['x'], eval_data['y'], eval_data['index'], f, batch_size, window_step)
  x_test, y_test = create_batches(test_data['x'], test_data['y'], test_data['index'], f, batch_size, window_step)

  print("x_batches: ", x_train.shape)
  print("y_batches: ", y_train.shape)

  return x_train, y_train, x_val, y_val, x_test, y_test, classes


def create_batches(x_data, y_data, index, f=32, batch_size=1, window_step=1):
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
    x_win = np.squeeze(view_as_windows(x_n, (m, f), step=window_step))

    # window length
    l_win = x_win.shape[0]

    # append y
    y = np.append(y, [y_n] * l_win)

    # stack windowed [n+r x m x f]
    x = np.vstack((x, x_win))

    #for i in range(r):
    #  plot_mfcc_only(x[i], fs, hop, plot_path, name=index[0] + str(i))

    # TODO: remove debug line later
    # i += 1
    # if i > 10:
    #  break

  # randomize examples
  indices = np.random.permutation(x.shape[0])
  x = np.take(x, indices, axis=0)
  y = np.take(y, indices, axis=0)

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



def train_nn(model, x_train, y_train, x_val, y_val, classes, nn_arch, num_epochs=2, lr=1e-3, model_path='./'):
  """
  train the neural network thing
  """

  # MSE Loss
  criterion = torch.nn.CrossEntropyLoss()

  # create optimizer
  optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.5)

  print("\n--Training starts:")

  # collect losses over epochs
  batch_loss = np.zeros(num_epochs)

  # start time
  start_time = time.time()

  # epochs
  for epoch in range(num_epochs):

    # cumulated loss
    cum_loss = 0.0

    # TODO: do this with loader function from pytorch
    # fetch data samples
    for i, (x, y) in enumerate(zip(x_train, y_train)):

      # zero parameter gradients
      optimizer.zero_grad()

      # prepare x [1 x 1 x 39 x 32]
      x = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(x.astype(np.float32)), 0), 0)
      #print("x: ", x.shape)

      # get index of class [b]
      y = get_index_of_class(y, classes, to_torch=True)

      # forward pass o:[b x c]
      o = model(x)

      # loss
      loss = criterion(o, y)

      # backward
      loss.backward()

      # optimizer step - update params
      optimizer.step()

      # loss update
      cum_loss += loss.item()

      # batch loss
      batch_loss[epoch] += cum_loss

      # print loss
      if i % 200 == 199:

        # print info
        print('epoch: {}, mini-batch: {}, loss: [{:.5f}]'.format(epoch + 1, i + 1, cum_loss / 10))
          
        # zero cum loss
        cum_loss = 0.0
    
  print('Training finished')

  # log time
  logging.info('Traning on arch: {} with examples: {}, n_classes: {}, num_epochs: {}, lr: {}, time: {}'.format(nn_arch, len(y_train), len(classes), num_epochs, lr, s_to_hms_str(time.time() - start_time)))

  # save parameters of network
  torch.save(model.state_dict(), model_path)

  return model, batch_loss


def eval_nn(model, x_batches, y_batches, classes):
  """
  evaluation of nn
  """

  # metric init
  correct, total = 0, 0

  # no gradients for eval
  with torch.no_grad():

    # load data
    for i, (x, y) in enumerate(zip(x_batches, y_batches)):

      # prepare x
      x = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(x.astype(np.float32)), 0), 0)

      # get index of class
      y = get_index_of_class(y, classes, to_torch=True)

      # classify
      o = model(x)

      # prediction
      _, y_hat = torch.max(o.data, 1)

      #print("pred: {}, actual: {}: ".format(y_hat, y))

      # add total amount of prediction
      total += y.size(0)

      # check if correctly predicted
      correct += (y_hat == y).sum().item()

  # print accuracy
  eval_log = 'Eval: correct: [{} / {}] acc: [{:.4f}]'.format(correct, total, 100 * correct / total)
  print(eval_log), logging.info(eval_log)


def get_nn_model(nn_arch):
  """
  simply get the desired nn model
  """

  # select network architecture
  if nn_arch == 'conv-trad':

    # traditional conv-net
    model = ConvNetTrad()

  else:

    # traditional conv-net
    model = ConvNetTrad()

  return model


def init_logging():
  """
  init logging stuff
  """

  logging.basicConfig(filename='./ignore/logs/ml.log', level=logging.DEBUG, format='%(asctime)s %(message)s')
  #logging.debug('This message should go to the log file')
  #logging.info('So should this')
  #logging.warning('And this, too')


if __name__ == '__main__':
  """
  ML - Machine Learning file
  """

  # path to train, test and eval set
  mfcc_data_files = ['./ignore/train/mfcc_data_train_n-100_c-5.npz', './ignore/test/mfcc_data_test_n-100_c-5.npz', './ignore/eval/mfcc_data_eval_n-100_c-5.npz']
  #mfcc_data_files = ['./ignore/train/mfcc_data_train_n-500_c-5.npz', './ignore/test/mfcc_data_test_n-500_c-5.npz', './ignore/eval/mfcc_data_eval_n-500_c-5.npz']

  # plot path and model path
  plot_path, model_path = './ignore/plots/ml/',  './ignore/models/'

  # create folder
  create_folder([plot_path, model_path])

  # other stuff
  init_logging()

  # frames for input
  f = 32
  batch_size = 1
  window_step = 5

  # extract all necessary data batches
  x_train, y_train, x_val, y_val, x_test, y_test, classes = extract_to_batches(mfcc_data_files, f=f, batch_size=batch_size, window_step=window_step)


  # -- names:
  # n: samples
  # l: length of frames 
  # m: features per frame
  # r: stride of frames 
  # f: frame length for input into NN
  #
  # -- shapes:
  # x:  [n x m x l]
  # xr: [n x m x r x f]


  # --
  # actual training

  # params for training
  num_epochs, lr, retrain = 1, 1e-4, True

  # nn architecture
  nn_architectures = ['conv-trad']

  # select architecture
  nn_arch = nn_architectures[0]

  # model name
  model_path = model_path + nn_arch + '_it-' + str(num_epochs) + '_lr-' + str(lr).replace('.', 'p') + '.pth'

  # init model
  model = get_nn_model(nn_arch)


  # check if model already exists
  if not os.path.exists(model_path) or retrain:

    # train
    model, loss = train_nn(model, x_train, y_train, x_val, y_val, classes, nn_arch, num_epochs=num_epochs, lr=lr, model_path=model_path)


  # load model params from file
  else:

    # load
    model.load_state_dict(torch.load(model_path))



  # --
  # evaluation

  # evaluation of model
  eval_nn(model, x_test, y_test, classes)



