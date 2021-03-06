"""
Insight file for investigating the nn models
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa

# my stuff
from common import create_folder
from plots import *
from conv_nets import *

from feature_extraction import FeatureExtractor, frames_to_sample
from classifier import Classifier

from skimage.util.shape import view_as_windows


if __name__ == '__main__':
  """
  Insight file - gives insight in neural nets
  """
  
  import yaml
  
  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # plot path and model path
  plot_path = './ignore/plots/insight/'

  # create folder
  create_folder([plot_path])

  # classifier
  classifier = Classifier(cfg_classifier=cfg['classifier'])


  # wav file to evaluate
  #wav = './ignore/my_recordings/clean_records/down.wav'
  wav = './ignore/my_recordings/clean_records/up.wav'


  # --
  # classification insights

  # init feature extractor
  feature_extractor = FeatureExtractor(cfg['feature_params'])

  # read audio from file
  x_raw, _ = librosa.load(wav, sr=cfg['feature_params']['fs'])

  # mfcc
  x_mfcc, _ = feature_extractor.extract_mfcc39(x_raw, reduce_to_best_onset=False)

  print("mfcc: ", x_mfcc.shape)

  # dimensions
  m, f, window_step = 39, 32, 2

  # windowed [r x m x f]
  x_win = np.squeeze(view_as_windows(x_mfcc, (m, f), step=window_step))
  y_hat_list = []

  for i, x in enumerate(x_win):

    # classify
    print("frame: ", i)
    y_hat, label = classifier.classify_sample(x)
    y_hat_list.append(y_hat)

    # plot
    time_s = frames_to_sample(i*window_step, cfg['feature_params']['fs'], feature_extractor.hop)
    time_e = frames_to_sample(i*window_step+32, cfg['feature_params']['fs'], feature_extractor.hop)

    plot_waveform(x_raw[time_s:time_e], cfg['feature_params']['fs'], title='frame{}_y-{}'.format(i, y_hat), xlim=None, ylim=(-1, 1), plot_path=plot_path, name='frame{}'.format(i))

  print("y_hats: ", y_hat_list)
  plot_waveform(x_raw, cfg['feature_params']['fs'], ylim=(-1, 1))

  plt.show()



