"""
choosing colormaps
"""

# qualitative colors
from palettable.colorbrewer.qualitative import Dark2_7

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


# diverging
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


# qualitative
from palettable.cartocolors.qualitative import Prism_8
from palettable.cartocolors.qualitative import Vivid_10
from palettable.cartocolors.qualitative import Antique_4

from palettable.colorbrewer.qualitative import Accent_4


# --
# global colors

hist_colors = Antique_4.mpl_colors

#reds = [red_16.mpl_colors[2], red_16.mpl_colors[5], red_16.mpl_colors[8], red_16.mpl_colors[11]]
reds4 = [red_16.mpl_colors[2], red_16.mpl_colors[5], red_16.mpl_colors[8], red_16.mpl_colors[11]]
reds4_2 = [red_16.mpl_colors[3], red_16.mpl_colors[5], red_16.mpl_colors[7], red_16.mpl_colors[10]]

colors_waveform = (None, Prism_8.mpl_colors[0:], Antique_4.mpl_colors[0:], purple_16.mpl_colors[::2][1:], reds4_2, reds4)

# cmaps
cmaps_weights = (None, red_16.mpl_colormap, purple_16.mpl_colormap, Deep_20.mpl_colormap.reversed(), jim_special_16.mpl_colormap, Delta_20.mpl_colormap)
#cmaps_weights = (None, red_16.mpl_colormap, Broc_19.mpl_colormap, Balance_19.mpl_colormap, Curl_19.mpl_colormap, Delta_19.mpl_colormap)

cmaps_mfcc = (None, Dense_20.mpl_colormap.reversed(), Tokyo_20.mpl_colormap, Deep_20.mpl_colormap.reversed(), red_16.mpl_colormap, jim_special_16.mpl_colormap)
#cmaps = (None, Dense_20.mpl_colormap.reversed(), purple_16.mpl_colormap, Deep_20.mpl_colormap.reversed(), red_16.mpl_colormap, jim_special_16.mpl_colormap)
#cmaps = (None, perceptual_rainbow_16.mpl_colormap, purple_16.mpl_colormap, cubehelix2_16.mpl_colormap, red_16.mpl_colormap, jim_special_16.mpl_colormap)
#cmaps = (None, Ice_20.mpl_colormap, Thermal_20.mpl_colormap, Deep_20.mpl_colormap.reversed(), Dense_20.mpl_colormap.reversed(), Matter_20.mpl_colormap.reversed())
#cmaps = (None, Ice_20.mpl_colormap, Matter_20.mpl_colormap.reversed(), Deep_20.mpl_colormap.reversed(), Dense_20.mpl_colormap.reversed(), Tempo_20.mpl_colormap.reversed())
#cmaps = (None, Ice_20.mpl_colormap, Matter_20.mpl_colormap.reversed(), Deep_20.mpl_colormap.reversed(), Dense_20.mpl_colormap.reversed(), Solar_20.mpl_colormap)
#cmaps = (None, Algae_20.mpl_colormap.reversed(), Amp_20.mpl_colormap.reversed(), Deep_20.mpl_colormap.reversed(), Dense_20.mpl_colormap.reversed(), Haline_20.mpl_colormap)


def show_hist_colors(x, colors):
  """
  histogram
  """

  for i, c in enumerate(colors):

    fig = plot_histogram(x, bins=None, color=c, y_log_scale=False, x_log_scale=False, context='None', title='', plot_path=None, name='hist', show_plot=False)
    
    # positioning
    if i >= 3: i, j = i%3, 600
    else: i, j = i, 0
    fig.canvas.manager.window.setGeometry(i*600, j, 600, 500)



def show_train_score_colors(x1, x2, x3, x4, cmaps):
  """
  waveform colors
  """

  for i, cmap in enumerate(cmaps):

    train_score = TrainScore(len(x1), is_adv=True)
    train_score.g_loss_fake, train_score.g_loss_sim, train_score.d_loss_real, train_score.d_loss_fake = x1, x2, x3, x4

    # plot weight matrices
    fig = plot_val_acc(x1, cmap=cmap)
    #fig = plot_train_loss(x1, x2, cmap=cmap)
    #fig = plot_adv_train_loss(train_score, cmap=cmap)
    
    # positioning
    if i >= 3: i, j = i%3, 600
    else: i, j = i, 0
    fig.canvas.manager.window.setGeometry(i*600, j, 600, 500)


def show_waveform_colors(x, cmaps):
  """
  waveform colors
  """

  for i, cmap in enumerate(cmaps):

    # plot weight matrices
    fig = plot_waveform(x, 16000, e=None, cmap=cmap, hop=None, onset_frames=None, title='none', xlim=None, ylim=None, plot_path=None, name='None', show_plot=False)
    
    # positioning
    if i >= 3: i, j = i%3, 600
    else: i, j = i, 0
    fig.canvas.manager.window.setGeometry(i*600, j, 600, 500)


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

  import torch
  import numpy as np
  import matplotlib.pyplot as plt
  import yaml
  import librosa

  import sys
  sys.path.append("../")

  from plots import plot_mfcc_only, plot_grid_images, plot_waveform, plot_val_acc, plot_train_loss, plot_adv_train_loss, plot_histogram
  from audio_dataset import AudioDataset
  from batch_archive import SpeechCommandsBatchArchive
  from net_handler import NetHandler
  from score import TrainScore

  # yaml config file
  cfg = yaml.safe_load(open("../config.yaml"))

  # audio sets
  audio_set1 = AudioDataset(cfg['datasets']['speech_commands'], cfg['feature_params'], root_path='../')
  audio_set2 = AudioDataset(cfg['datasets']['my_recordings'], cfg['feature_params'], root_path='../')

  # create batches
  batch_archive = SpeechCommandsBatchArchive(feature_file_dict={**audio_set1.feature_file_dict, **audio_set2.feature_file_dict}, batch_size_dict={'train': 32, 'test': 5, 'validation': 5, 'my': 1}, shuffle=False)

  # create batches of selected label
  batch_archive.create_batches()

  # net handler
  net_handler = NetHandler(nn_arch='conv-jim', class_dict=batch_archive.class_dict, data_size=batch_archive.data_size, use_cpu=cfg['ml']['use_cpu'])

  # load model files
  #net_handler.load_models(model_files=['/world/cavern/git/kws_game/not_relevant/ignore/models/bs-32_it-1077_lr-0p0001/cnn_model.pth'])
  #net_handler.load_models(model_files=['./ignore/models/bs-32_it-500_lr-0p0001/cnn_model.pth'])
  #print(net_handler.models['cnn'].state_dict().keys())

  # get some examples
  x1 = batch_archive.x_train[0, 0, 0]
  x2 = batch_archive.x_train[0, 1, 0]

  print("x1: ", x1.shape)
  print("x2: ", x2.shape)
  print("x1: ", batch_archive.z_train[0, 0])
  print("x2: ", batch_archive.z_train[0, 1])

  n1 = torch.randn(x1.shape)
  n2 = torch.randn(x2.shape)

  # read audio from file
  x_wav, _ = librosa.load('../ignore/my_recordings/clean_records/down.wav', sr=16000)

  x_sine1 = np.sin(np.linspace(0, 2 * np.pi, 100)) * 10 + 50
  x_sine2 = np.sin(np.linspace(0, 2 * np.pi * 2, 100)) * 10 + 50
  x_sine3 = np.sin(np.linspace(0, 2 * np.pi * 3, 100)) * 10 + 50
  x_sine4 = np.sin(np.linspace(0, 2 * np.pi * 4, 100)) * 10 + 50

  # mfcc colormaps
  #show_mfcc_colormaps(x1, cmaps_mfcc)
  #show_weights_colormaps(net_handler.models['cnn'].state_dict()['conv_encoder.conv_layers.0.weight'], cmaps_weights)
  #show_weights_colormaps(net_handler.models['cnn'].state_dict()['conv_encoder.conv_layers.1.weight'], cmaps_weights)
  #show_waveform_colors(x_wav, colors_waveform)
  show_train_score_colors(x_sine1, x_sine2, x_sine3, x_sine4, colors_waveform)
  #show_hist_colors(x_sine1, hist_colors)


  # vmax = np.max(np.abs(x_wav))

  # plt.figure()
  # ax = plt.axes()
  # #ax.set_prop_cycle('color', Vivid_10.mpl_colors[:2])
  # #ax.set_prop_cycle('color', Prism_8.mpl_colors)
  # #ax.plot(x_wav, color=Prism_8.mpl_colors[0])
  # ax.plot(x_wav, color=Antique_4.mpl_colors[0])
  # plt.ylim(vmax + 0.1 * vmax, -vmax - 0.1 * vmax)
  # #ax.plot(x_wav, color=red_16.mpl_colors[5])
  # for i in range(10):
  #   #ax.plot(np.ones(len(x_wav)) * i * 0.005)

  #   #ax.axvline(x=2000*i, dashes=(5, 1), color=Prism_8.mpl_colors[0])
  #   ax.axvline(x=2000*i, dashes=(5, 1), color=red_16.mpl_colors[13])
  #   #ax.axvline(x=2100*i, dashes=(5, 1), color=Antique_4.mpl_colors[0])
  #   #ax.axvline(x=1000*i, dashes=(5, 1), color=Accent_4.mpl_colors[0])

  # plt.grid()

  plt.show()



