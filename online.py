"""
Online Microphone input classification
"""

import numpy as np
import matplotlib.pyplot as plt

import os
import torch

# my stuff
from common import *
from plots import *
from conv_nets import *

from ml import get_nn_model



if __name__ == '__main__':
  """
  ML - Machine Learning file
  """

  # path to model
  model_path = './ignore/models/'

  # plot path and model path
  #plot_path, shift_path, metric_path, model_path = './ignore/plots/ml/', './ignore/plots/ml/shift/', './ignore/plots/ml/metrics/', './ignore/models/'
  # create folder
  #create_folder([plot_path, shift_path, metric_path, model_path])

  # var selection for model
  nn_arch, version_id, n_examples_class, batch_size, num_epochs, lr = 'conv-trad', 2, 2000, 32, 200, 1e-4

  # init model
  model = get_nn_model(nn_arch)

  # param string
  param_str = '{}_v{}_n-{}_bs-{}_it-{}_lr-{}'.format(nn_arch, version_id, n_examples_class, batch_size, num_epochs, str(lr).replace('.', 'p'))

  # load model
  model.load_state_dict(torch.load(model_path + param_str + '.pth'))
  print(model)


  # --
  # read mic input

  plt.show()



