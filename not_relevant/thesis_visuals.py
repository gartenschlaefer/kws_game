"""
thesis visualizations
"""

import numpy as np
import librosa
import re
import os
from glob import glob

import sys
sys.path.append("../")

from audio_dataset import AudioDataset
from feature_extraction import FeatureExtractor, custom_dct_matrix
from plots import plot_mel_band_weights, plot_mfcc_profile, plot_waveform, plot_dct, plot_wav_grid, plot_spec_profile, plot_mel_scale
from latex_table_maker import LatexTableMaker


def get_infos_from_log(in_file):
  """
  get infos from log
  """

  # init
  infos, training_line = {}, ''

  # get training info line
  with open(in_file, "r") as f:
    for line in f:
      if 'Training on arch:' in line:
        training_line = line
        break

  # sub params
  dataset_param_string = re.sub(r'(param string: )|([\[\] \/])', '', re.findall(r'param string: \[[\w\-\_0-9\/]+\]', training_line)[0])
  train_params = re.sub(r'train_params: ', '', re.findall(r'train_params: {[\w\-\_0-9\/\:\., \']+}', training_line)[0])

  # extract from training line
  infos.update({'arch':re.sub(r'(arch: )|([\[\]])', '', re.findall(r'arch: \[[\w-]+\]', training_line)[0])})
  infos.update({'L':re.sub(r'c-', '', re.findall(r'c-[0-9]+', dataset_param_string)[0])})
  infos.update({'n':re.sub(r'n-', '', re.findall(r'n-[0-9]+', dataset_param_string)[0])})
  infos.update({'bs':re.sub(r'\'batch_size\': ', '', re.findall(r'\'batch_size\': [0-9]+', train_params)[0])})
  infos.update({'it':re.sub(r'\'num_epochs\': ', '', re.findall(r'\'num_epochs\': [0-9]+', train_params)[0])})
  infos.update({'lr':re.sub(r'\'lr\': ', '', re.findall(r'\'lr\': [0-9\.]+', train_params)[0])})
  infos.update({'mo':re.sub(r'\'beta\': ', '', re.findall(r'\'beta\': [0-9\.]+', train_params)[0])})

  return infos


def get_thesis_table_captions(in_file):
  """
  collection of captions
  """

  # get param infos
  info = get_infos_from_log(in_file)

  # edit caption
  caption = 'Feature Selection on arch: {} with dataset: L{}-n{}'.format(info['arch'], info['L'], info['n'])
  caption += ' and training params: it{}-bs{}-lr{}-mo{}'.format(info['it'], info['bs'], info['lr'], info['mo'])

  return caption


def mfcc_stuff(cfg):
  """
  for dct, filter bands, etc
  """

  # plot path
  plot_path = '../docu/thesis/3_signal/figs/'

  # init feature extractor
  feature_extractor = FeatureExtractor(cfg['feature_params'])

  # plot dct
  plot_dct(custom_dct_matrix(cfg['feature_params']['n_filter_bands']), plot_path=plot_path, name='signal_mfcc_dct', show_plot=False)
  plot_dct(custom_dct_matrix(cfg['feature_params']['n_filter_bands']), plot_path=plot_path, context='dct-div', name='signal_mfcc_dct-div', show_plot=False)

  # mel scale
  plot_mel_scale(plot_path=plot_path, name='signal_mfcc_mel_scale', show_plot=False)

  # plot mel bands
  plot_mel_band_weights(feature_extractor.w_f, feature_extractor.w_mel, feature_extractor.f, feature_extractor.m, plot_path=plot_path, name='signal_mfcc_weights', show_plot=True)


def showcase_wavs(cfg, raw_plot=True, spec_plot=True, mfcc_plot=True, show_plot=False):
  """
  showcase wavs
  """

  # plot path
  plot_path = '../docu/thesis/3_signal/figs/'

  # change params
  feature_params = cfg['feature_params'].copy()
  feature_params['n_ceps_coeff'] = 32
  feature_params['norm_features'] = True

  # init feature extractor
  feature_extractor = FeatureExtractor(feature_params)

  # wav, anno dir
  wav_dir, anno_dir = '../ignore/my_recordings/showcase_wavs/', '../ignore/my_recordings/showcase_wavs/annotation/'

  # analyze some wavs
  for wav, anno in zip(glob(wav_dir + '*.wav'), glob(anno_dir + '*.TextGrid')):

    # info
    print("\nwav: ", wav), print("anno: ", anno)

    # load file
    x, _ = librosa.load(wav, sr=feature_params['fs'])

    # raw waveform
    if raw_plot: plot_waveform(x, feature_params['fs'], anno_file=anno, hop=feature_extractor.hop, plot_path=plot_path, name='signal_raw_' + wav.split('/')[-1].split('.')[0] + '_my', show_plot=show_plot)
    
    # spectogram
    if spec_plot: 
      plot_spec_profile(x, feature_extractor.calc_spectogram(x).T, feature_params['fs'], feature_extractor.N, feature_extractor.hop, anno_file=anno, plot_path=plot_path, title=wav.split('/')[-1].split('.')[0]+'_my', name='signal_spec-lin_' + wav.split('/')[-1].split('.')[0] + '_my', show_plot=show_plot)
      plot_spec_profile(x, feature_extractor.calc_spectogram(x).T, feature_params['fs'], feature_extractor.N, feature_extractor.hop, log_scale=True, anno_file=anno, plot_path=plot_path, title=wav.split('/')[-1].split('.')[0]+'_my', name='signal_spec-log_' + wav.split('/')[-1].split('.')[0] + '_my', show_plot=show_plot)

    # mfcc
    if mfcc_plot:
      mfcc, bon_pos = feature_extractor.extract_mfcc(x, reduce_to_best_onset=False)
      plot_mfcc_profile(x, cfg['feature_params']['fs'], feature_extractor.N, feature_extractor.hop, mfcc, anno_file=anno, sep_features=True, bon_pos=bon_pos, frame_size=cfg['feature_params']['frame_size'], plot_path=plot_path, name='signal_mfcc_' + wav.split('/')[-1].split('.')[0] + '_my', close_plot=False, show_plot=show_plot)


def feature_selection_tables(overwrite=False):
  """
  feature selection tables
  """

  # files
  in_files = ['../ignore/logs/ml_it1000_c5_features_trad.log', '../ignore/logs/ml_it1000_c5_features_fstride.log', '../ignore/logs/ml_it2000_c30_features_fc3.log', '../ignore/logs/ml_it1000_c30_features_fc1.log', '../ignore/logs/ml_it500_c5_features_fc1.log']
  out_files = ['../docu/thesis/4_practice/tables/tab_fs_trad_it1000_c5.tex', '../docu/thesis/4_practice/tables/tab_fs_fstride_it1000_c5.tex', '../docu/thesis/4_practice/tables/tab_fs_fc3_it2000_c30.tex', '../docu/thesis/4_practice/tables/tab_fs_fc1_it1000_c30.tex', '../docu/thesis/4_practice/tables/tab_fs_fc1_it500_c5.tex']

  for in_file, out_file in zip(in_files, out_files):

    # check files existance
    if os.path.isfile(out_file) and not overwrite:
      print("out file exists: ", out_file)
      continue

    # table info
    print("feature selection table: ", out_file)

    # instances
    lt_maker = LatexTableMaker(in_file=in_file, extraction_type='feature_selection')

    # extract table
    tables = lt_maker.extract_table(out_file=out_file, caption=get_thesis_table_captions(in_file))


def audio_set_wavs(cfg):
  """
  audio set wavs
  """

  # plot path
  plot_path = '../docu/thesis/5_exp/figs/'

  # audio sets
  a1 = AudioDataset(cfg['datasets']['speech_commands'], cfg['feature_params'], root_path='../')
  a2 = AudioDataset(cfg['datasets']['my_recordings'], cfg['feature_params'], root_path='../')

  # feature extractor
  feature_extractor = FeatureExtractor(cfg['feature_params'])

  # get audio files
  a1.get_audiofiles()

  # random seed
  np.random.seed(1234)
  r = np.random.randint(low=0, high=150, size=len(a1.set_audio_files[1]))

  wav_grid = []

  # process wavs
  for wav in sorted([label_wavs[r[i]] for i, label_wavs in enumerate(a1.set_audio_files[1])]):

    # info
    print("wav: ", wav)

    # get raw
    x, _ = a1.wav_pre_processing(wav)

    # extract feature vectors [m x l]
    _, bon_pos = feature_extractor.extract_mfcc(x, reduce_to_best_onset=False)

    # append to wav grid
    wav_grid.append((librosa.util.normalize(x), re.sub(r'[0-9]+-', '', wav.split('/')[-1].split('.')[0]), bon_pos))

  # plot wav grid
  plot_wav_grid(wav_grid, feature_params=a1.feature_params, grid_size=(6, 5), plot_path=plot_path, name='wav_grid_c30', show_plot=True)



if __name__ == '__main__':
  """
  main
  """

  import yaml

  # yaml config file
  cfg = yaml.safe_load(open("../config.yaml"))

  # mfcc stuff
  #mfcc_stuff(cfg)

  # showcase wavs
  #showcase_wavs(cfg, raw_plot=True, spec_plot=True, mfcc_plot=True, show_plot=False)
  showcase_wavs(cfg, raw_plot=False, spec_plot=False, mfcc_plot=True, show_plot=True)

  # feature selection tables
  #feature_selection_tables(overwrite=True)

  # audio set wavs
  #audio_set_wavs(cfg)


