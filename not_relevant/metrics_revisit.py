"""
revisit metrics
"""

import numpy as np

import sys
sys.path.append("../")

from plots import plot_train_score
from pathlib import Path


if __name__ == '__main__':
  """
  main
  """

  import yaml

  # yaml config file
  cfg = yaml.safe_load(open("../config.yaml"))

  # metric path
  #model_path = '../ignore/models/hyb-jim/v5_c7n1m1_n-500_r1-5_mfcc32-12_c1d0d0e0_norm1_f-1x12x50/bs-32_it-1000_lr-d-0p0001_lr-g-0p0001/'
  model_path = '../docu/models/ignore/exp_cepstral/'

  # select models
  model_sel = ['conv-fstride', 'conv-jim', 'conv-trad']

  # model dictionary
  model_dict = {'{}'.format(ms): [{'model': m, 'params': p, 'metrics': me} for i, (m, p, me) in enumerate(zip(Path(model_path).rglob('*.pth'), Path(model_path).rglob('params.npz'), Path(model_path).rglob('metrics.npz'))) if str(m).find(ms) != -1] for ms in model_sel}

  # load metrics
  metrics = np.load(model_dict['conv-fstride'][0]['metrics'], allow_pickle=True)

  # see whats in data
  print(metrics.files)

  train_score_dict = metrics['train_score_dict'][()]

  print("train_score_dict: ", train_score_dict.keys())
  print("score_class: ", train_score_dict['score_class'])

  # plot train score
  plot_train_score(train_score_dict, plot_path=None, name_ext='_revisit', show_plot=True)

