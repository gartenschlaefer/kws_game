"""
delete files, take care
"""

import os
import numpy as np
from pathlib import Path


if __name__ == '__main__':
  """
  main
  """

  # path to delete
  #model_path = '../docu/best_models/exp_cepstral/'
  #model_path = '../docu/best_models/exp_mfcc/'
  #model_path = '../docu/best_models/exp_adv_label/'
  model_path = '../docu/best_models/exp_adv_train/'
  #model_path = '../docu/best_models/exp_wavenet/'
  #model_path = '../docu/best_models/exp_adv_label/'

  # get all png
  pngs = Path(model_path).rglob('*.png')

  # info scores
  info_scores = Path(model_path).rglob('info_score.txt')

  # dirs
  diff_plots_dir = Path(model_path).rglob('diff_plots/')
  conv_plots_dir = Path(model_path).rglob('conv_plots/')
  train_col_dir = Path(model_path).rglob('train_collections/')


  # delete files
  [(print(p), os.remove(p)) for p in pngs]
  [(print(p), os.remove(p)) for p in info_scores]

  # delete directories
  [(print(p), os.rmdir(p)) for p in diff_plots_dir]
  [(print(p), os.rmdir(p)) for p in conv_plots_dir]
  [(print(p), os.rmdir(p)) for p in train_col_dir]
  
