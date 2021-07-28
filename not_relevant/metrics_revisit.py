"""
revisit metrics
"""

import numpy as np

import sys
sys.path.append("../")

from plots import plot_train_score
from score import TrainScore

if __name__ == '__main__':
  """
  main
  """

  import yaml

  # yaml config file
  cfg = yaml.safe_load(open("../config.yaml"))

  # metric path
  metric_path = '../ignore/models/hyb-jim/v5_c7n1m1_n-500_r1-5_mfcc32-12_c1d0d0e0_norm1_f-1x12x50/bs-32_it-1000_lr-d-0p0001_lr-g-0p0001/'

  # load metrics
  metrics = np.load(metric_path + cfg['ml']['metrics_file_name'], allow_pickle=True)

  # see whats in data
  print(metrics.files)

  train_score_dict = metrics['train_score_dict'][()]

  print("train_score_dict: ", train_score_dict.keys())

  # plot train score
  plot_train_score(train_score_dict, plot_path=metric_path, name_ext='_revisit')

