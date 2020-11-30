"""
extract examples from recorded files in .wav format
"""

import numpy as np
import matplotlib.pyplot as plt

import re
import librosa
import yaml
import soundfile

from glob import glob

# my stuff
from feature_extraction import calc_onsets

from plots import *
from audio_dataset import extract_mfcc_data
from path_collector import PathCollector


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


def cut_and_copy_wavs(wavs, params, wav_path, plot_path, recut=True):
  """
  runs through each wav and gets onsets of individual examples
  """

  # list of labels
  label_list = np.array([])

  # windowing samples
  N, hop = int(params['N_s'] * params['fs']), int(params['hop_s'] * params['fs'])

  # get all wav files and save them
  for i, wav in enumerate(wavs):

    print("wav: ", wav)

    # extract filename
    file_name = re.findall(r'[\w+ 0-9]+\.wav', wav)[0]

    # extract label from filename
    label = re.sub(r'(_v[0-9]+\.wav)', '', file_name)
    label_list = np.append(label_list, label)

    if recut == False:
      print("***recut disabled!!!")
      continue

    # read audio from file
    x, _ = librosa.load(wav, sr=params['fs'])

    # calc onsets
    onsets = calc_onsets(x, params['fs'], N=N, hop=hop, adapt_frames=5, adapt_alpha=0.09, adapt_beta=0.8)
    onsets = clean_onsets(onsets, frame_length=params['frame_size']*5)

    # cut examples to one second
    x_cut = cut_signal(x, params['fs'], onsets, hop, time=1, alpha=0.4)

    # plot onsets
    plot_onsets(x, params['fs'], N, hop, onsets, title=label, plot_path=plot_path, name='onsets_{}'.format(label))

    for j, xj in enumerate(x_cut):
      
      # plot
      plot_waveform(xj, params['fs'], title='{}-{}'.format(label, j), plot_path=plot_path, name='example_{}-{}'.format(label, j))

      # save file
      soundfile.write('{}{}{}.wav'.format(wav_path, label, j), xj, params['fs'], subtype=None, endian=None, format=None, closefd=True)

  return np.unique(label_list)


if __name__ == '__main__':
  """
  reads recorded examples from in_path, cuts them to single examples and saves them
  """

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # path_collector
  path_coll = PathCollector(cfg)

  # create all necessary folders
  path_coll.create_my_recording_folders()


  # --
  # cut

  # get all .wav files
  raw_wavs = glob(cfg['my_recordings']['in_path'] + '*.wav')

  # cut them to single wavs
  labels = cut_and_copy_wavs(raw_wavs, cfg['feature_params'], cfg['my_recordings']['wav_path'], cfg['my_recordings']['plot_path'], recut=cfg['my_recordings']['recut'])


  # --
  # extract mfcc features

  # all wavs init
  all_wavs = []

  # get all wavs from selected labels
  for l in cfg['my_recordings']['sel_labels']:

    # wav regex
    wav_name_re = '*' + l + '[0-9]*.wav'

    # get wavs
    wavs = glob(cfg['my_recordings']['wav_path'] + wav_name_re)

    # add to all wavs
    all_wavs.append(wavs)

    # check length of labels
    print("overall stat of label: [{}]\tnum: [{}]".format(l, len(wavs)))

  # extract features
  mfcc_data, label_data, index_data = extract_mfcc_data(wavs=all_wavs, params=cfg['feature_params'], n_examples=cfg['my_recordings']['n_examples'], set_name=cfg['my_recordings']['set_name'], plot_path_mfcc=cfg['my_recordings']['plot_path'], plot_path_z_score=cfg['my_recordings']['plot_path'], enable_plot=cfg['my_recordings']['enable_plot'])

  # save mfcc data file
  np.savez(path_coll.mfcc_data_file_my, x=mfcc_data, y=label_data, index=index_data, params=cfg['feature_params'])


  # prints
  print("--save data to: ", path_coll.mfcc_data_file_my)
  print("n_examples: {}, labels: {}".format(cfg['my_recordings']['n_examples'], cfg['my_recordings']['sel_labels']))
  print("mfcc_data: ", mfcc_data.shape)

  plt.show()








