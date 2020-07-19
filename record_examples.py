"""
Machine Learning file for training and evaluating the model
"""

import numpy as np
import matplotlib.pyplot as plt

import re
import librosa

from glob import glob

# my stuff
from feature_extraction import calc_onsets
from common import *
from plots import *
from audio_dataset import extract_mfcc_data


def clean_onsets(onsets, frame_length=32):
  """
  clean onsets, so that only one per examples exists
  """

  # save old onsets
  onsets = onsets.copy()

  # init
  clean_onsets = np.zeros(onsets.shape)

  # setup primitive onset filter
  onset_filter = np.zeros(frame_length)
  onset_filter[0] = 1

  # onset filtering
  for i, onset in enumerate(onsets):

    # stop at end
    if i > len(onsets) - frame_length // 2:
      break

    # clean
    if onset:
      onsets[i:i+frame_length] = onset_filter
      clean_onsets[i] = 1

  return clean_onsets


def cut_signal(x, fs, onsets, hop, time=1, alpha=0.5):
  """
  cut signal at onsets to a specific time interval
  """

  # init cut signal
  x_cut = np.zeros((int(np.sum(onsets)), int(fs * time)))

  # post factor
  beta = 1 - alpha

  # amount of samples 
  num_samples = fs / time

  # pre and post samples
  pre = int(alpha * num_samples)
  post = int(beta * num_samples)

  # onset index
  oi = 0

  for i, onset in enumerate(onsets):

    if onset:
      x_cut[oi, :] = x[i*hop-pre:i*hop+post]
      oi += 1

  return x_cut


def cut_and_copy_wavs(wavs, wav_path, plot_path, recut=True):
  """
  runs through each wav and gets onsets of individual examples
  """

  label_list = np.array([])

  # get all wav files and save them
  for i, wav in enumerate(wavs):

    print("wav: ", wav)

    # extract filename
    file_name = re.findall(r'[\w+ 0-9]+\.wav', wav)[0]

    # extract label from filename
    label = re.sub(r'(_v[0-9]+\.wav)', '', file_name)
    label_list = np.append(label_list, label)

    if recut == False:
      continue

    # read audio from file
    x, _ = librosa.load(wav, sr=fs)

    # calc onsets
    onsets = calc_onsets(x, fs, N=N, hop=hop, adapt_frames=5, adapt_alpha=0.09, adapt_beta=0.8)
    onsets = clean_onsets(onsets, frame_length=32*2)

    # cut examples to one second
    x_cut = cut_signal(x, fs, onsets, hop, time=1, alpha=0.4)

    # plot onsets
    plot_onsets(x, fs, N, hop, onsets, title=label, plot_path=plot_path, name='onsets_{}'.format(label))

    for j, xj in enumerate(x_cut):
      
      # plot
      plot_waveform(xj, fs, title='{}-{}'.format(label, j), plot_path=plot_path, name='example_{}-{}'.format(label, j))

      # save file
      librosa.output.write_wav('{}{}{}.wav'.format(wav_path, label, j), xj, fs)

  return np.unique(label_list)


if __name__ == '__main__':
  """
  get examples from recordings
  """

  # root path
  root_path = './ignore/my_recordings/'

  # plot path and model path
  in_path, plot_path, wav_path = root_path + 'raw/', root_path + 'plots/', root_path + 'wav/'

  # create folder
  create_folder([plot_path, wav_path])

  # extension for file name
  ext = 'my'
  version_nr = 2

  # --
  # params

  # sampling frequency
  fs = 16000

  # mfcc / onset window and hop size
  N, hop = int(0.025 * fs), int(0.010 * fs)

  # amount of filter bands and cepstral coeffs
  n_filter_bands, n_ceps_coeff = 32, 12


  # --
  # cut

  # get all .wav files
  raw_wavs = glob(in_path + '*.wav')

  # cut them to single wavs
  labels = cut_and_copy_wavs(raw_wavs, wav_path, plot_path, recut=False)


  # --
  # extract mfcc features

  # get all wavs
  wavs = glob(wav_path + '*.wav')
  n_examples = len(wavs)

  # params container and info string
  params = {'n_examples':n_examples, 'data_percs':[1], 'fs':fs, 'N':N, 'hop':hop, 'n_filter_bands':n_filter_bands, 'n_ceps_coeff':n_ceps_coeff}
  info = "n_examples={} with data split {}, fs={}, mfcc: N={} is t={}, hop={} is t={}, n_f-bands={}, n_ceps_coeff={}".format(n_examples, [1], fs, N, N/fs, hop, hop/fs, n_filter_bands, n_ceps_coeff)

  # extract features
  mfcc_data, label_data, index_data = extract_mfcc_data(wavs, params, frame_size=32, ext=ext, plot_path=plot_path)

  # set file name
  file_name = '{}mfcc_data_{}_n-{}_c-{}_v{}.npz'.format(root_path, ext, n_examples, len(labels), version_nr)

  # save mfcc data file
  np.savez(file_name, x=mfcc_data, y=label_data, index=index_data, info=info, params=params)

  # print
  print("--save data to: ", file_name)

  print("n_examples: {}, labels: {}".format(n_examples, labels))
  print("mfcc_data: ", mfcc_data.shape)

  plt.show()








