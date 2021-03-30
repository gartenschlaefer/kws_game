"""
choosing colormaps
"""

import numpy as np
import torch

import matplotlib.pyplot as plt

import sys
sys.path.append("../")

import yaml
from plots import plot_mfcc_only, plot_grid_images
from audio_dataset import AudioDataset
from batch_archive import SpeechCommandsBatchArchive
from net_handler import NetHandler

# qualitative colors
from palettable.colorbrewer.qualitative import Dark2_7

# --
# mfcc:

# sequential colors
#from palettable.colorbrewer.sequential import Purples_4
#from palettable.cartocolors.sequential import agSunset_7
#from palettable.cartocolors.sequential import agGrnYl_7

#from palettable.cmocean.sequential import Algae_20
#from palettable.cmocean.sequential import Amp_20
from palettable.cmocean.sequential import Deep_20
from palettable.cmocean.sequential import Dense_20
#from palettable.cmocean.sequential import Haline_20
#from palettable.cmocean.sequential import Ice_20
#from palettable.cmocean.sequential import Matter_20
#from palettable.cmocean.sequential import Solar_20
#from palettable.cmocean.sequential import Speed_20
#from palettable.cmocean.sequential import Tempo_20
#from palettable.cmocean.sequential import Thermal_20
#from palettable.cmocean.sequential import Turbid_20

from palettable.cubehelix import classic_16
#from palettable.cubehelix import cubehelix1_16
#from palettable.cubehelix import cubehelix2_16
#from palettable.cubehelix import cubehelix3_16
from palettable.cubehelix import jim_special_16
from palettable.cubehelix import perceptual_rainbow_16
from palettable.cubehelix import purple_16
from palettable.cubehelix import red_16

from palettable.scientific.sequential import Tokyo_20


# --
# weights

# diverging colors
from palettable.cmocean.diverging import Balance_20
from palettable.cmocean.diverging import Balance_19
from palettable.cmocean.diverging import Curl_20
from palettable.cmocean.diverging import Delta_20

from palettable.scientific.diverging import Broc_20
from palettable.scientific.diverging import Cork_20
#from palettable.scientific.diverging import Vik_20
from palettable.scientific.diverging import Berlin_5

from palettable.colorbrewer.diverging import PuOr_5
from palettable.colorbrewer.diverging import PuOr_11


# --
# global cmaps

# cmaps
cmaps_weights = (None, red_16.mpl_colormap, purple_16.mpl_colormap, Cork_20.mpl_colormap, Balance_19.mpl_colormap, Delta_20.mpl_colormap)
#cmaps_weights = (None, red_16.mpl_colormap, Broc_19.mpl_colormap, Balance_19.mpl_colormap, Curl_19.mpl_colormap, Delta_19.mpl_colormap)

cmaps_mfcc = (None, Dense_20.mpl_colormap.reversed(), Tokyo_20.mpl_colormap, Deep_20.mpl_colormap.reversed(), red_16.mpl_colormap, jim_special_16.mpl_colormap)
#cmaps = (None, Dense_20.mpl_colormap.reversed(), purple_16.mpl_colormap, Deep_20.mpl_colormap.reversed(), red_16.mpl_colormap, jim_special_16.mpl_colormap)
#cmaps = (None, perceptual_rainbow_16.mpl_colormap, purple_16.mpl_colormap, cubehelix2_16.mpl_colormap, red_16.mpl_colormap, jim_special_16.mpl_colormap)
#cmaps = (None, Ice_20.mpl_colormap, Thermal_20.mpl_colormap, Deep_20.mpl_colormap.reversed(), Dense_20.mpl_colormap.reversed(), Matter_20.mpl_colormap.reversed())
#cmaps = (None, Ice_20.mpl_colormap, Matter_20.mpl_colormap.reversed(), Deep_20.mpl_colormap.reversed(), Dense_20.mpl_colormap.reversed(), Tempo_20.mpl_colormap.reversed())
#cmaps = (None, Ice_20.mpl_colormap, Matter_20.mpl_colormap.reversed(), Deep_20.mpl_colormap.reversed(), Dense_20.mpl_colormap.reversed(), Solar_20.mpl_colormap)
#cmaps = (None, Algae_20.mpl_colormap.reversed(), Amp_20.mpl_colormap.reversed(), Deep_20.mpl_colormap.reversed(), Dense_20.mpl_colormap.reversed(), Haline_20.mpl_colormap)


def show_weights_colormaps(x, cmaps):
  """
  colormaps of weights
  """

  for i, cmap in enumerate(cmaps):

    # plot weight matrices
    fig = plot_grid_images(x, padding=1, num_cols=8, cmap=cmap, color_balance=True, show_plot=False)

    # positioning
    if i >= 3: i, j = i%3, 600
    else: i, j = i, 0
    fig.canvas.manager.window.setGeometry(i*600, j, 600, 500)


def show_mfcc_colormaps(x, cmaps):
  """
  mfcc colormaps
  """

  for i, cmap in enumerate(cmaps):

    # image config
    fig = plt.figure()
    ax = plt.axes()
    #ax.set_prop_cycle('color', Dark2_7.mpl_colors)

    # image
    #im = ax.imshow(x, cmap=cmap, vmax=vmax, vmin=-vmax)
    im = ax.imshow(x, cmap=cmap)

    # colorbar
    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    plt.colorbar(im, cax=cax)

    # move windows
    if i >= 3: i, j = i%3, 600
    else: i, j = i, 0
    fig.canvas.manager.window.move(i*640, j)



if __name__ == '__main__':
  """
  color tests
  """

  # yaml config file
  cfg = yaml.safe_load(open("../config.yaml"))

  # audio sets
  audio_set1 = AudioDataset(cfg['datasets']['speech_commands'], cfg['feature_params'], root_path='../')
  audio_set2 = AudioDataset(cfg['datasets']['my_recordings'], cfg['feature_params'], root_path='../')

  # create batches
  batch_archive = SpeechCommandsBatchArchive(audio_set1.feature_files + audio_set2.feature_files, batch_size=32, batch_size_eval=5)

  # net handler
  net_handler = NetHandler(nn_arch='conv-encoder', class_dict=batch_archive.class_dict, data_size=batch_archive.data_size, use_cpu=cfg['ml']['use_cpu'])

  # load model files
  net_handler.load_models(model_files=['/world/cavern/git/kws_game/not_relevant/ignore/models/bs-32_it-1077_lr-0p0001/cnn_model.pth'])

  print(net_handler.models['cnn'].state_dict().keys())


  # get some examples
  x1 = batch_archive.x_train[0, 0, 0]
  x2 = batch_archive.x_train[0, 1, 0]

  print("x1: ", x1.shape)
  print("x2: ", x2.shape)
  print("x1: ", batch_archive.z_train[0, 0])
  print("x2: ", batch_archive.z_train[0, 1])

  n1 = torch.randn(x1.shape)
  n2 = torch.randn(x2.shape)

  # mfcc colormaps
  #show_mfcc_colormaps(x1, cmaps_mfcc)
  show_weights_colormaps(net_handler.models['cnn'].state_dict()['conv_encoder.conv_layers.0.weight'], cmaps_weights)
  #show_weights_colormaps(net_handler.models['cnn'].state_dict()['conv_encoder.conv_layers.1.weight'], cmaps_weights)

  plt.show()



