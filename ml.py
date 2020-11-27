"""
Machine Learning file for training and evaluating the model
"""

import numpy as np
import matplotlib.pyplot as plt

import logging
import time

import os
import torch
import yaml

from skimage.util.shape import view_as_windows

# my stuff
from common import *
from plots import *
from net_handler import CnnHandler


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
  init_logging(cfg['ml']['log_path'])


  # create batches
  batch_archiv = BatchArchiv(cfg['ml']['mfcc_data_files'], batch_size=cfg['ml']['train_params']['batch_size'])

  # params
  #params = {'version_id':version_id, 'f':f, 'batch_size':batch_size, 'num_epochs':num_epochs, 'lr':lr, 'nn_arch':nn_arch, 'pre_trained_model_path':pre_trained_model_path}


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
  param_str = '{}_v{}_c-{}_n-{}_bs-{}_it-{}_lr-{}'.format(cfg['ml']['nn_arch'], cfg['audio_dataset']['version_nr'], len(batch_archiv.classes), batch_archiv.n_examples_class, cfg['ml']['train_params']['batch_size'], cfg['ml']['train_params']['num_epochs'], str(cfg['ml']['train_params']['lr']).replace('.', 'p'))


  # model handler
  cnn_handler = CnnHandler(nn_arch=cfg['ml']['nn_arch'], batch_archiv=batch_archiv) 

  # load pre trained model
  if cfg['ml']['use_pre_trained_model']:
    cnn_handler.load_pre_trained_model(cfg['ml']['pre_trained_model_path'])


  # --
  # training

  # check if model already exists
  if not os.path.exists(cfg['ml']['model_path'] + param_str + '.pth') or cfg['ml']['retrain']:

    # train
    score_collecotr = cnn_handler.train_nn(num_epochs=cfg['ml']['train_params']['num_epochs'], lr=cfg['ml']['train_params']['lr'], param_str=param_str)

    # training info
    logging.info('Traning on arch: {}  time: {}'.format(param_str, s_to_hms_str(score_collector.time_usage)))
    
    # save model
    cnn_handler.save_model(cfg['ml']['model_path'] + param_str + '.pth')

    # save as pre trained model as well
    if cfg['ml']['save_model_as_pre_model']:
      cnn_handler.save_model('{}{}_c-{}{}'.format(cfg['ml']['pre_trained_model_path'], cnn_handler.nn_arch, len(cnn_handler.batch_archiv.classes), '.pth'))

    # save infos
    np.savez(cfg['ml']['model_path'] + param_str + '.npz', params=params, param_str=param_str, class_dict=batches.class_dict, model_file_path=model_path + param_str + '.pth')
    np.savez(cfg['ml']['metric_path'] + 'metrics_' + param_str + '.npz', score_collector=score_collector)

    # plots
    plot_train_loss(score_collector.train_loss, score_collector.val_loss, cfg['ml']['plot_path'], name=param_str + '_train_loss')
    plot_val_acc(score_collector.val_acc, cfg['ml']['plot_path'], name=param_str + '_val_acc')

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
  cnn_handler.model.eval()

  # evaluation of model
  eval_score = cnn_handler.eval_nn(cnn_handler.model, calc_cm=True)

  # print accuracy
  eval_log = eval_score.info_log()


  # --
  # info output

  # log to file
  if logging_enabled:
    logging.info(eval_log)

  # print confusion matrix
  print("confusion matrix:\n", eval_score.cm)

  # plot confusion matrix
  plot_confusion_matrix(eval_score.cm, batch_archiv.classes, plot_path=cfg['ml']['plot_path'], name=param_str + '_confusion_test')


  # --
  # evaluation on my set
  if x_my is not None:

    print("\n--Evaluation on My Set:")

    # evaluation of model
    eval_score = eval_nn(model, x_my, y_my, classes, z_batches=batch_archiv.z_my, calc_cm=True, verbose=True)
    print("confusion matrix:\n", eval_score.cm)

    # plot confusion matrix
    plot_confusion_matrix(eval_score.cm, batch_archiv.classes, plot_path=cfg['ml']['plot_path'], name=param_str + '_confusion_my')


  # show all plots
  plt.show()



