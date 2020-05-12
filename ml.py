"""
Machine Learning file for training and evaluating the model
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage.util.shape import view_as_windows

from common import create_folder
from plots import plot_mfcc_only

def prepare_data(x, y, index, f):
  """
  prepare data for training [r x m x f]
  """

  # TODO: implemenation
  pass


if __name__ == '__main__':
  """
  ML - Machine Learning file
  """

  # path to train, test and eval set
  mfcc_data_files = ['./ignore/train/mfcc_data_train_n-100_c-5.npz', './ignore/test/mfcc_data_test_n-100_c-5.npz', './ignore/eval/mfcc_data_eval_n-100_c-5.npz']

  # plot path
  plot_path = './ignore/plots/ml/'

  # create folder
  create_folder([plot_path])

  # load files
  data = [np.load(file, allow_pickle=True) for file in mfcc_data_files]

  # training data
  train_data = data[0]

  print("train_data: ", train_data.files)

  # extract data from file
  x_data, y, index, info, params = train_data['x'], train_data['y'], train_data['index'], train_data['info'], train_data['params']

  # shape of things
  n, m, l = x_data.shape

  # get some params
  fs, hop = params[()]['fs'], params[()]['hop']

  #print("x_data: ", x_data.shape)
  #print("index: ", index)

  # frames for input
  f = 32

  # -- names:
  # n: samples
  # l: length of frames 
  # m: features per frame
  # r: stride of frames, 
  # f: frame length for input into NN
  #
  # -- shapes:
  # x:  [n x m x l]
  # xr: [n x m x r x f]

  # --
  # prepare x for training

  # plot mfcc only
  #plot_mfcc_only(x_data[0], fs, hop, plot_path, name=index[0])


  #x, y = prepare_data()

  # run through all samples
  for x_n in x_data:

    # windowed [r x m x f]
    x = np.squeeze(view_as_windows(x_n, (m, f), step=1))

    # TODO: Only use meaning ful vectors not noise

    print("xn: ", x_n.shape)
    print("x: ", x.shape)

    r = x.shape[0]

    #for i in range(r):
    #  plot_mfcc_only(x[i], fs, hop, plot_path, name=index[0] + str(i))

    break
