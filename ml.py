"""
Machine Learning file for training and evaluating the model
"""

import numpy as np
import matplotlib.pyplot as plt

import logging

import os
import yaml

# my stuff
from common import s_to_hms_str
from path_collector import PathCollector
from plots import plot_val_acc, plot_train_loss, plot_confusion_matrix
from batch_archive import SpeechCommandsBatchArchive
from net_handler import CnnHandler, AdversarialNetHandler


def init_logging(log_path):
  """
  init logging stuff
  """

  # config
  logging.basicConfig(filename=log_path + 'ml.log', level=logging.DEBUG, format='%(asctime)s %(message)s')

  # disable unwanted logs
  logging.getLogger('matplotlib.font_manager').disabled = True


if __name__ == '__main__':
  """
  ML - Machine Learning file
  """

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # init path collector
  path_coll = PathCollector(cfg)

  # create all necessary folders
  path_coll.create_ml_folders()


  # init logging
  init_logging(cfg['ml']['paths']['log'])


  # --
  # batches

  # create batch archiv
  batch_archive = SpeechCommandsBatchArchive(path_coll.mfcc_data_files_all, batch_size=cfg['ml']['train_params']['batch_size'])

  # print classes
  print("classes: ", batch_archive.classes)


  # --
  # model handler

  # create model handler
  if cfg['ml']['nn_arch'] == 'adv-experimental':
    nn_handler = AdversarialNetHandler(nn_arch=cfg['ml']['nn_arch'], use_cpu=cfg['ml']['use_cpu'])

  else:
    nn_handler = CnnHandler(nn_arch=cfg['ml']['nn_arch'], n_classes=batch_archive.n_classes, use_cpu=cfg['ml']['use_cpu']) 

  # load pre trained model
  if cfg['ml']['load_pre_model']:
    nn_handler.load_model(path_coll=path_coll, for_what='pre')


  # --
  # training

  # check if model already exists
  if not os.path.exists(path_coll.model_file) or cfg['ml']['retrain']:

    # train
    train_score = nn_handler.train_nn(train_params=cfg['ml']['train_params'], batch_archive=batch_archive)

    # training info
    logging.info('Traning on arch: [{}], train_params: {}, device: [{}], time: {}'.format(cfg['ml']['nn_arch'], cfg['ml']['train_params'], nn_handler.device, s_to_hms_str(train_score.time_usage)))
    
    # save model
    nn_handler.save_model(path_coll=path_coll, train_params=cfg['ml']['train_params'], class_dict=batch_archive.class_dict, train_score=train_score, save_as_pre_model=cfg['ml']['save_as_pre_model'])

    # plots
    plot_train_loss(train_score.train_loss, train_score.val_loss, plot_path=path_coll.model_path, name='train_loss')
    plot_val_acc(train_score.val_acc, plot_path=path_coll.model_path, name='val_acc')

  # load model params from file without training
  else:

    # load model
    nn_handler.load_model(path_coll=path_coll, for_what='trained')


  # --
  # evaluation on test set

  print("\n--Evaluation on Test Set:")

  # activate eval mode (no dropout layers)
  nn_handler.set_eval_mode()

  # evaluation of model
  eval_score = nn_handler.eval_nn(eval_set='test', batch_archive=batch_archive, calc_cm=True, verbose=False)

  # print accuracy
  eval_log = eval_score.info_log(do_print=False)


  # --
  # info output

  # log to file
  if cfg['ml']['logging_enabled']:
    logging.info(eval_log)

  # print confusion matrix
  print("confusion matrix:\n{}\n".format(eval_score.cm))

  # plot confusion matrix
  plot_confusion_matrix(eval_score.cm, batch_archive.classes, plot_path=path_coll.model_path, name='confusion_test')


  # --
  # evaluation on my set
  if batch_archive.x_my is not None:

    print("\n--Evaluation on My Set:")

    # evaluation of model
    eval_score = nn_handler.eval_nn(eval_set='my', batch_archive=batch_archive, calc_cm=True, verbose=True)
    print("confusion matrix:\n{}\n".format(eval_score.cm))

    # plot confusion matrix
    plot_confusion_matrix(eval_score.cm, batch_archive.classes, plot_path=path_coll.model_path, name='confusion_my')


  # show all plots
  #plt.show()



