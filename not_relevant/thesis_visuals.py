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
from latex_table_maker import LatexTableMaker, LatexTableMakerAudiosetLabels


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


def mfcc_stuff(cfg, dct_plot=False, mel_scale_plot=False, mel_band_plot=False, show_plot=False):
  """
  for dct, filter bands, etc
  """

  # plot path
  plot_path = '../docu/thesis/3_signal/figs/'

  # init feature extractor
  feature_extractor = FeatureExtractor(cfg['feature_params'])

  # plot dct
  if dct_plot:
    plot_dct(custom_dct_matrix(N=32, C=12), plot_path=plot_path, name='signal_mfcc_dct12', show_plot=show_plot)
    plot_dct(custom_dct_matrix(N=32, C=12), plot_path=plot_path, context='dct-div', name='signal_mfcc_dct12-div', show_plot=show_plot)

  # mel scale
  if mel_scale_plot: plot_mel_scale(plot_path=plot_path, name='signal_mfcc_mel_scale', show_plot=show_plot)

  # plot mel bands
  if mel_band_plot: plot_mel_band_weights(feature_extractor.w_f, feature_extractor.w_mel, feature_extractor.f, feature_extractor.m, plot_path=plot_path, name='signal_mfcc_weights', show_plot=show_plot)


def showcase_wavs(cfg, raw_plot=True, spec_plot=True, mfcc_plot=True, use_mfcc_39=False, show_plot=False):
  """
  showcase wavs
  """

  # plot path
  plot_path = '../docu/thesis/3_signal/figs/'

  # change params
  feature_params = cfg['feature_params'].copy()
  feature_params['n_ceps_coeff'] = 12 if use_mfcc_39 else 32
  feature_params['use_delta_features'] = True if use_mfcc_39 else False
  feature_params['use_double_delta_features'] = True if use_mfcc_39 else False
  feature_params['use_energy_features'] = True if use_mfcc_39 else False
  feature_params['norm_features'] = True

  # init feature extractor
  feature_extractor = FeatureExtractor(feature_params)

  # wav, anno dir
  #wav_dir, anno_dir = '../ignore/my_recordings/showcase_wavs/', '../ignore/my_recordings/showcase_wavs/annotation/'
  wav_dir, anno_dir = '../docu/showcase_wavs/', '../docu/showcase_wavs/annotation/'

  # analyze some wavs
  for wav, anno in zip(glob(wav_dir + '*.wav'), glob(anno_dir + '*.TextGrid')):

    # info
    print("\nwav: ", wav), print("anno: ", anno)

    # load file
    x, _ = librosa.load(wav, sr=feature_params['fs'])

    # raw waveform
    if raw_plot: plot_waveform(x, feature_params['fs'], anno_file=anno, hop=feature_extractor.hop, plot_path=plot_path, name='signal_raw_showcase_' + wav.split('/')[-1].split('.')[0], show_plot=show_plot)
    
    # spectogram
    if spec_plot:
      plot_spec_profile(x, feature_extractor.calc_spectogram(x).T, feature_params['fs'], feature_extractor.N, feature_extractor.hop, anno_file=anno, plot_path=plot_path, name='signal_spec-lin_showcase_' + wav.split('/')[-1].split('.')[0], show_plot=show_plot)
      plot_spec_profile(x, feature_extractor.calc_spectogram(x).T, feature_params['fs'], feature_extractor.N, feature_extractor.hop, log_scale=True, anno_file=anno, plot_path=plot_path, name='signal_spec-log_showcase_' + wav.split('/')[-1].split('.')[0], show_plot=show_plot)

    # mfcc
    if mfcc_plot:
      mfcc, bon_pos = feature_extractor.extract_mfcc(x, reduce_to_best_onset=False)
      name = 'signal_mfcc_showcase_mfcc32_' + wav.split('/')[-1].split('.')[0] if not use_mfcc_39 else 'signal_mfcc_showcase_mfcc39_' + wav.split('/')[-1].split('.')[0]
      plot_mfcc_profile(x, cfg['feature_params']['fs'], feature_extractor.N, feature_extractor.hop, mfcc, anno_file=anno, sep_features=False, bon_pos=bon_pos, frame_size=cfg['feature_params']['frame_size'], plot_path=plot_path, name=name, close_plot=False, show_plot=show_plot)


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


def audio_set_wavs(cfg, statistics_plot=True, wav_grid_plot=False, label_table_plot=False):
  """
  audio set wavs
  """

  # plot path
  plot_path = '../docu/thesis/5_exp/figs/'
  plot_path_tab = '../docu/thesis/5_exp/tables/'

  # audioset init
  audio_set1 = AudioDataset(cfg['datasets']['speech_commands'], feature_params=cfg['feature_params'], root_path='../')
  audio_set2 = AudioDataset(cfg['datasets']['my_recordings'], feature_params=cfg['feature_params'], root_path='../')

  # get audio files
  audio_set1.get_audiofiles()
  audio_set2.get_audiofiles()

  # statistics (saves them in dataset folder)
  if statistics_plot: audio_set1.analyze_dataset_extraction(calculate_overall_stats=True)

  # label table
  if label_table_plot: LatexTableMakerAudiosetLabels(audio_set1.all_label_file_dict, caption='all labels', label='tab:exp_dataset_all_labels', out_file=plot_path_tab + 'tab_exp_dataset_all_labels__not_released.tex')

  # plot wav grid
  if wav_grid_plot: 
    plot_wav_grid(audio_set1.extract_wav_examples(set_name='test', n_examples=1, from_selected_labels=False), feature_params=audio_set1.feature_params, grid_size=(6, 6), plot_path=plot_path, name='exp_dataset_wav_grid_speech_commands_v2', show_plot=True)
    plot_wav_grid(audio_set2.extract_wav_examples(set_name='my', n_examples=5), feature_params=audio_set2.feature_params, grid_size=(5, 5), plot_path=plot_path, name='exp_dataset_wav_grid_my', show_plot=True)



if __name__ == '__main__':
  """
  main
  """

  import yaml

  # yaml config file
  cfg = yaml.safe_load(open("../config.yaml"))

  # mfcc stuff
  #mfcc_stuff(cfg, dct_plot=True, show_plot=True)

  # showcase wavs
  #showcase_wavs(cfg, raw_plot=False, spec_plot=False, mfcc_plot=True, use_mfcc_39=True, show_plot=True)

  # feature selection tables
  #feature_selection_tables(overwrite=True)

  # audio set wavs
  audio_set_wavs(cfg, statistics_plot=True, wav_grid_plot=False, label_table_plot=False)


