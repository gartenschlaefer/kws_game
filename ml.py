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
from batch_archiv import BatchArchiv
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

  # init path collector
  path_coll = PathCollector(cfg)

  # create all necessary folders
  path_coll.create_ml_folders()


  # init logging
  init_logging(cfg['ml']['paths']['log'])


  # --
  # batches

  # create batch archiv
  batch_archiv = BatchArchiv(path_coll.mfcc_data_files_all, batch_size=cfg['ml']['train_params']['batch_size'])

  # print classes
  print("classes: ", batch_archiv.classes)


  # --
  # model handler

  # create model handler
  cnn_handler = CnnHandler(nn_arch=cfg['ml']['nn_arch'], n_classes=batch_archiv.n_classes, model_file_name=cfg['ml']['model_file_name'], use_cpu=cfg['ml']['use_cpu']) 

  # load pre trained model
  if cfg['ml']['load_pre_model']:
    cnn_handler.load_model(model_file=path_coll.model_pre_file)


  # --
  # training

  # check if model already exists
  if not os.path.exists(path_coll.model_file) or cfg['ml']['retrain']:

    # train
    train_score = cnn_handler.train_nn(train_params=cfg['ml']['train_params'], batch_archiv=batch_archiv)

    # training info
    logging.info('Traning on arch: [{}], train_params: {}, device: [{}], time: {}'.format(cfg['ml']['nn_arch'], cfg['ml']['train_params'], cnn_handler.device, s_to_hms_str(train_score.time_usage)))
    
    # save model
    cnn_handler.save_model(model_file=path_coll.model_file, params_file=path_coll.params_file, train_params=cfg['ml']['train_params'], class_dict=batch_archiv.class_dict, metric_file=path_coll.metrics_file, train_score=train_score, model_pre_file=path_coll.model_pre_file, save_as_pre_model=cfg['ml']['save_as_pre_model'])

    # plots
    plot_train_loss(train_score.train_loss, train_score.val_loss, plot_path=path_coll.model_path, name='train_loss')
    plot_val_acc(train_score.val_acc, plot_path=path_coll.model_path, name='val_acc')

  # load model params from file without training
  else:

    # load model
    cnn_handler.load_model(model_file=path_coll.model_file)

    # save infos
    np.savez(path_coll.model_path + cfg['ml']['params_file_name'], train_params=cfg['ml']['train_params'], class_dict=batch_archiv.class_dict, model_file=path_coll.model_file)


  # --
  # evaluation on test set

  print("\n--Evaluation on Test Set:")

  # activate eval mode (no dropout layers)
  cnn_handler.model.eval()

  # evaluation of model
  eval_score = cnn_handler.eval_nn(eval_set='test', batch_archiv=batch_archiv, calc_cm=True, verbose=False)

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
  plot_confusion_matrix(eval_score.cm, batch_archiv.classes, plot_path=path_coll.model_path, name='confusion_test')


  # --
  # evaluation on my set
  if batch_archiv.x_my is not None:

    print("\n--Evaluation on My Set:")

    # evaluation of model
    eval_score = cnn_handler.eval_nn(eval_set='my', batch_archiv=batch_archiv, calc_cm=True, verbose=True)
    print("confusion matrix:\n{}\n".format(eval_score.cm))

    # plot confusion matrix
    plot_confusion_matrix(eval_score.cm, batch_archiv.classes, plot_path=path_coll.model_path, name='confusion_my')


  # show all plots
  #plt.show()



