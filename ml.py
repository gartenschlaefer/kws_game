"""
Machine Learning file for training and evaluating the model
"""

import numpy as np
import matplotlib.pyplot as plt

import os
import torch
import yaml

from skimage.util.shape import view_as_windows

# my stuff
from common import *
from plots import *
from conv_nets import *

import logging
import time








def train_nn(model, x_train, y_train, x_val, y_val, classes, nn_arch, num_epochs=2, lr=1e-3, param_str='nope'):
  """
  train the neural network thing
  """

  # Loss Criterion
  criterion = torch.nn.CrossEntropyLoss()

  # create optimizer
  #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.5)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
      cum_loss = print_train_info(epoch, i, cum_loss, k_print=y_train.shape[0] // 10)

    # valdiation
    val_loss[epoch], val_acc[epoch], _ = eval_nn(model, x_val, y_val, classes, logging_enabled=False)

    # TODO: Early stopping if necessary
    
  print('--Training finished')

  # log time
  logging.info('Traning on arch: {}  time: {}'.format(param_str, s_to_hms_str(time.time() - start_time)))

  return model, train_loss, val_loss, val_acc


def eval_nn(model, x_batches, y_batches, classes, z_batches=None, logging_enabled=True, calc_cm=False, verbose=False):
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
      if verbose:
        if z_batches is not None:
          print("\nlabels: {}".format(z_batches[i]))
        print("output: {}\npred: {}\nactu: {}, \t corr: {} ".format(o.data, y_hat, y, (y_hat == y).sum().item()))

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









def init_logging(log_path):
  """
  init logging stuff
  """

  logging.basicConfig(filename=log_path + 'ml.log', level=logging.DEBUG, format='%(asctime)s %(message)s')

  # disable unwanted logs
  logging.getLogger('matplotlib.font_manager').disabled = True


if __name__ == '__main__':
  """
  ML - Machine Learning file
  """

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # create folder
  create_folder([cfg['ml']['plot_path'], cfg['ml']['shift_path'], cfg['ml']['metric_path'], cfg['ml']['model_path'], cfg['ml']['model_pre_path'], cfg['ml']['log_path']])

  # init logging
  init_logging(log_path)


  # create batches
  batch_archiv = BatchArchiv(cfg['ml']['mfcc_data_files'], batch_size=cfg['ml']['batch_size'])

  # --
  # version
  # 1: better onset detection
  # 2: energy frame onset detection

  # params for training
  num_epochs, lr, retrain = 500, 1e-5, True


  # select architecture
  nn_arch = nn_architectures[1]

  # pretrained model
  #pre_trained_model_path = model_pre_path + 'conv-fstride_c-30.pth'
  pre_trained_model_path = None


  # params
  #params = {'version_id':version_id, 'f':f, 'batch_size':batch_size, 'num_epochs':num_epochs, 'lr':lr, 'nn_arch':nn_arch, 'pre_trained_model_path':pre_trained_model_path}


  # extract all necessary data batches
  #x_train, y_train, x_val, y_val, x_test, y_test, x_my, y_my, z_my, classes, n_examples_class = extract_to_batches(cfg['ml']['mfcc_data_files'], f=f, batch_size=batch_size)

  print("classes: ", batch_archiv.classes)


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
  param_str = '{}_v{}_c-{}_n-{}_bs-{}_it-{}_lr-{}'.format(nn_arch, cfg['audio_dataset']['version_nr'], len(batch_archiv.classes), batch_archiv.n_examples_class, cfg['ml']['batch_size'], cfg['ml']['num_epochs'], str(lr).replace('.', 'p'))


  if pre_trained_model_path is not None:
    param_str = param_str + '_pre'


  # model trainer


  # init model
  model = get_nn_model(nn_arch, len(classes))

  # use pre-trained model
  model = get_pretrained_model(model, pre_trained_model_path)

  # --
  # training

  # check if model already exists
  if not os.path.exists(cfg['ml']['model_path'] + param_str + '.pth') or retrain:

    # train
    model, train_loss, val_loss, val_acc = train_nn(model, x_train, y_train, x_val, y_val, classes, nn_arch, num_epochs=cfg['ml']['num_epochs'], lr=cfg['ml']['lr'], param_str=param_str)

    # save model
    torch.save(model.state_dict(), model_path + param_str + '.pth')
    if pre_trained_model_path is not None:
      torch.save(model.state_dict(), '{}{}_c-{}{}'.format(model_pre_path, nn_arch, len(classes), '.pth'))

    # save infos
    np.savez(model_path + param_str + '.npz', params=params, param_str=param_str, class_dict=batches.class_dict, model_file_path=model_path + param_str + '.pth')
    np.savez(metric_path + 'metrics_' + param_str + '.npz', train_loss=train_loss, val_loss=val_loss, val_acc=val_acc)

    # plots
    plot_train_loss(train_loss, val_loss, plot_path, name=param_str + '_train_loss')
    plot_val_acc(val_acc, plot_path, name=param_str + '_val_acc')

  # load model params from file
  else:

    # load
    model.load_state_dict(torch.load(cfg['ml']['model_path'] + param_str + '.pth'))

    # save infos
    np.savez(cfg['ml']['model_path'] + param_str + '.npz', params=params, param_str=param_str, class_dict=batches.class_dict, model_file_path=cfg['ml']['model_path'] + param_str + '.pth')


  # --
  # evaluation on test set

  print("\n--Evaluation on Test Set:")

  # activate eval mode (no dropout layers)
  model.eval()

  # evaluation of model
  eval_loss, acc, cm = eval_nn(model, x_test, y_test, classes, calc_cm=True)
  print("confusion matrix:\n", cm)

  # plot confusion matrix
  plot_confusion_matrix(cm, classes, plot_path=plot_path, name=param_str + '_confusion_test')


  # --
  # evaluation on my set
  if x_my is not None:

    print("\n--Evaluation on My Set:")

    # evaluation of model
    eval_loss, acc, cm = eval_nn(model, x_my, y_my, classes, z_batches=z_my, calc_cm=True, verbose=True)
    print("confusion matrix:\n", cm)

    # plot confusion matrix
    plot_confusion_matrix(cm, classes, plot_path=plot_path, name=param_str + '_confusion_my')

  plt.show()



