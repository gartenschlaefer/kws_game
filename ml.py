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
from plots import *
from conv_nets import *

import logging
import time


def extract_to_batches(mfcc_data_files, f=32, batch_size=4, window_step=16):
  """
  extract data samples from files
  """

  # status message
  print("\n--extract batches from data")

  # load files [0]: train, etc.
  data = [np.load(file, allow_pickle=True) for file in mfcc_data_files]

  # extract data
  train_data, test_data, eval_data = data[0], data[1], data[2]
  #print("Container of train_data: ", train_data.files)

  # print some infos about data
  print("train: ", train_data['x'].shape), print("test: ", test_data['x'].shape), print("eval: ", eval_data['x'].shape)
  print("data info: ", train_data['info'])

  # get classes
  classes = np.unique(train_data['y'])
  print("classes: ", classes)

  # examples per class
  n_examples_class = (len(train_data['x']) + len(test_data['x']) + len(eval_data['x'])) // len(classes)

  # create batches
  x_train, y_train = create_batches(train_data, classes, f, batch_size=batch_size, window_step=window_step, plot_shift=False)
  x_val, y_val = create_batches(eval_data, classes, f, batch_size=4, window_step=window_step)
  x_test, y_test = create_batches(test_data, classes, f, batch_size=4, window_step=window_step)

  # print training batches
  print("x_train_batches: ", x_train.shape)
  print("y_train_batches: ", y_train.shape)

  return x_train, y_train, x_val, y_val, x_test, y_test, classes, n_examples_class


def create_batches(data, classes, f=32, batch_size=1, window_step=1, plot_shift=False):
  """
  create batches for training N x [b x m x f]
  x: [n x m x l]
  y: [n]
  N: Amount of batches
  b: batch size
  m: feature size
  f: frame length
  """

  # extract data
  x_data, y_data, index, params = data['x'], data['y'], data['index'], data['params']

  # get shape of things
  n, m, l = x_data.shape

  # randomize data
  indices = np.random.permutation(x_data.shape[0])
  x_data = np.take(x_data, indices, axis=0)
  y_data = np.take(y_data, indices, axis=0)

  # x: [n x m x f]
  x = np.empty(shape=(0, 39, f), dtype=x_data.dtype)
  y = np.empty(shape=(0), dtype=y_data.dtype)

  # stack windows
  for i, (x_n, y_n) in enumerate(zip(x_data, y_data)):

    # windowed [r x m x f]
    x_win = np.squeeze(view_as_windows(x_n, (m, f), step=window_step))

    # window length
    l_win = x_win.shape[0]

    # append y
    y = np.append(y, [y_n] * l_win)

    # stack windowed [n+r x m x f]
    x = np.vstack((x, x_win))

    # for evaluation of shifting
    if i < 2 and plot_shift:

      # need params for plot
      fs, hop = params[()]['fs'], params[()]['hop']
      index = np.take(index, indices, axis=0)

      # plot example
      plot_mfcc_only(x_n, fs, hop, shift_path, name='{}-{}'.format(index[i], i))

      # plot some shifted mfcc
      for ri in range(x_win.shape[0]):

        # plot shifted mfcc
        plot_mfcc_only(x[ri], fs, hop, shift_path, name='{}-{}-{}'.format(index[i], i, ri))

    # #TODO: remove debug line later
    # i += 1
    # if i > 5:
    #  break


  # randomize examples
  indices = np.random.permutation(x.shape[0])
  x = np.take(x, indices, axis=0)
  y = np.take(y, indices, axis=0)

  # number of windows
  batch_nums = x.shape[0] // batch_size

  # remaining samples
  r = int(np.remainder(len(y), batch_size))
  if r:
    batch_nums += 1;

  # init batches
  x_batches = torch.empty((batch_nums, batch_size, 39, f))
  y_batches = torch.empty((batch_nums, batch_size), dtype=torch.long)

  # batching
  for i in range(batch_nums):

    # remainder handling
    if i == batch_nums - 1 and r:

      # remaining examples
      r_x = x[i*batch_size:i*batch_size+r, :]
      r_y = y[i*batch_size:i*batch_size+r]

      # pick random samples for filler
      random_samples = np.random.randint(0, high=len(y), size=batch_size-r)

      # filling examples
      f_x = x[random_samples, :]
      f_y = y[random_samples]

      # concatenate remainder with random examples
      x_batches[i, :] = torch.from_numpy(np.concatenate((r_x, f_x)).astype(np.float32))
      y_batches[i, :] = get_index_of_class(np.concatenate((r_y, f_y)), classes, to_torch=True)

    # no remainder
    else:

      # get batches
      x_batches[i, :] = torch.from_numpy(x[i*batch_size:i*batch_size+batch_size, :].astype(np.float32))
      y_batches[i, :] = get_index_of_class(y[i*batch_size:i*batch_size+batch_size], classes, to_torch=True)

  # prepare for training x: [num_batches x batch_size x channel x 39 x 32]
  x_batches = torch.unsqueeze(x_batches, 2)

  return x_batches, y_batches


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

  # init labels
  y_idx = torch.empty(y.shape, dtype=torch.long)

  for i, yi in enumerate(y):

    # get index
    idx = np.where(np.array(classes) == yi)[0]

    # transfer to torch
    if to_torch:
      y_idx[i] = torch.from_numpy(idx)

  return y_idx


def train_nn(model, x_train, y_train, x_val, y_val, classes, nn_arch, num_epochs=2, lr=1e-3, param_str='nope'):
  """
  train the neural network thing
  """

  # Loss Criterion
  criterion = torch.nn.CrossEntropyLoss()

  # create optimizer
  optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.5)

  print("\n--Training starts:")

  # collect losses over epochs
  train_loss = np.zeros(num_epochs)
  val_loss = np.zeros(num_epochs)
  val_acc = np.zeros(num_epochs)

  # start time
  start_time = time.time()

  # epochs
  for epoch in range(num_epochs):

    # cumulated loss
    cum_loss = 0.0

    # TODO: do this with loader function from pytorch (maybe or not)
    # fetch data samples
    for i, (x, y) in enumerate(zip(x_train, y_train)):

      # zero parameter gradients
      optimizer.zero_grad()

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
      train_loss[epoch] += cum_loss

      # print some infos, reset cum_loss
      cum_loss = print_train_info(epoch, i, cum_loss, k_print=10)

    # valdiation
    val_loss[epoch], val_acc[epoch], _ = eval_nn(model, x_val, y_val, classes, logging_enabled=False)

    # TODO: Early stopping if necessary
    
  print('--Training finished')

  # log time
  logging.info('Traning on arch: {}  time: {}'.format(param_str, s_to_hms_str(time.time() - start_time)))

  return model, train_loss, val_loss, val_acc


def eval_nn(model, x_batches, y_batches, classes, logging_enabled=True, calc_cm=False):
  """
  evaluation of nn
  """

  # Loss Criterion
  criterion = torch.nn.CrossEntropyLoss()

  # init
  correct, total, eval_loss, cm = 0, 0, 0.0, None
  y_all, y_hat_all = np.empty(shape=(0), dtype=y_batches.numpy().dtype), np.empty(shape=(0), dtype=y_batches.numpy().dtype)

  # no gradients for eval
  with torch.no_grad():

    # load data
    for i, (x, y) in enumerate(zip(x_batches, y_batches)):

      # classify
      o = model(x)

      # loss
      eval_loss += criterion(o, y)

      # prediction
      _, y_hat = torch.max(o.data, 1)

      # add total amount of prediction
      total += y.size(0)

      # check if correctly predicted
      correct += (y_hat == y).sum().item()

      # collect labels for confusion matrix
      if calc_cm:
        y_all = np.append(y_all, y)
        y_hat_all = np.append(y_hat_all, y_hat)

      # some prints
      #print("\npred: {}\nactu: {}, \t corr: {} ".format(y_hat, y, (y_hat == y).sum().item()))

  # print accuracy
  eval_log = "Eval: correct: [{} / {}] acc: [{:.4f}] with loss: [{:.4f}]\n".format(correct, total, 100 * correct / total, eval_loss)
  print(eval_log)

  # confusion matrix
  if calc_cm:
    cm = confusion_matrix(y_all, y_hat_all)

  # log to file
  if logging_enabled:
    logging.info(eval_log)

  return eval_loss, (correct / total), cm


def print_train_info(epoch, mini_batch, cum_loss, k_print=10):
  """
  print some training info
  """

  # print loss
  if mini_batch % k_print == k_print-1:

    # print info
    print('epoch: {}, mini-batch: {}, loss: [{:.5f}]'.format(epoch + 1, mini_batch + 1, cum_loss / k_print))

    # zero cum loss
    cum_loss = 0.0

  return cum_loss



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

  # disable unwanted logs
  logging.getLogger('matplotlib.font_manager').disabled = True

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
  plot_path, shift_path, metric_path, model_path = './ignore/plots/ml/', './ignore/plots/ml/shift/', './ignore/plots/ml/metrics/', './ignore/models/'

  # create folder
  create_folder([plot_path, shift_path, metric_path, model_path])

  # init logging
  init_logging()

  # batch stuff
  f, batch_size, window_step = 32, 32, 16

  # params for training
  num_epochs, lr, retrain = 25, 1e-4, False

  # nn architecture
  nn_architectures = ['conv-trad']

  # select architecture
  nn_arch = nn_architectures[0]



  # extract all necessary data batches
  x_train, y_train, x_val, y_val, x_test, y_test, classes, n_examples_class = extract_to_batches(mfcc_data_files, f=f, batch_size=batch_size, window_step=window_step)


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


  # param string
  param_str = '{}_n-{}_ws-{}_bs-{}_it-{}_lr-{}'.format(nn_arch, n_examples_class, window_step, batch_size, num_epochs, str(lr).replace('.', 'p'))

  # model name
  model_path = model_path + param_str + '.pth'

  # init model
  model = get_nn_model(nn_arch)


  # --
  # training

  # check if model already exists
  if not os.path.exists(model_path) or retrain:

    # train
    model, train_loss, val_loss, val_acc = train_nn(model, x_train, y_train, x_val, y_val, classes, nn_arch, num_epochs=num_epochs, lr=lr, param_str=param_str)

    # save model
    torch.save(model.state_dict(), model_path)

    # save metrics
    np.savez(metric_path + 'metrics_' + param_str + '.npz', train_loss=train_loss, val_loss=val_loss, val_acc=val_acc)

    # plots
    plot_train_loss(train_loss, val_loss, plot_path, name='train_loss_' + param_str)
    plot_val_acc(val_acc, plot_path, name='val_acc_' + param_str)

  # load model params from file
  else:

    # load
    model.load_state_dict(torch.load(model_path))


  # --
  # evaluation

  print("\n--Evaluation on Test Set:")

  # evaluation of model
  eval_loss, acc, cm = eval_nn(model, x_test, y_test, classes, calc_cm=True)

  print("confusion matrix:\n", cm)

  # plot confusion matrix
  plot_confusion_matrix(cm, classes, plot_path=plot_path, name='confusion_' + param_str)


  plt.show()



