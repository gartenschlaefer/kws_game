"""
thesis visualizations
"""

import torch
import numpy as np
import librosa
import re
import os
from glob import glob

import sys
sys.path.append("../")

from test_bench import TestBench
from audio_dataset import AudioDataset
from feature_extraction import FeatureExtractor, custom_dct_matrix
from batch_archive import SpeechCommandsBatchArchive
from plots import plot_mel_band_weights, plot_mfcc_profile, plot_waveform, plot_dct, plot_wav_grid, plot_spec_profile, plot_mel_scale, plot_grid_images, plot_mfcc_plain, plot_activation_function
from latex_table_maker import LatexTableMakerMFCC, LatexTableMakerAudiosetLabels, LatexTableMakerCepstral, LatexTableMakerAdv, LatexTableMakerFinal
from skimage.util.shape import view_as_windows


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
    plot_dct(custom_dct_matrix(N=32, C=32).T, plot_path=plot_path, name='signal_mfcc_dct', show_plot=show_plot)
    plot_dct(custom_dct_matrix(N=32, C=32).T, plot_path=plot_path, context='dct-div', name='signal_mfcc_dct-div', show_plot=show_plot)

  # mel scale
  if mel_scale_plot: plot_mel_scale(plot_path=plot_path, name='signal_mfcc_mel_scale', show_plot=show_plot)

  # plot mel bands
  if mel_band_plot: plot_mel_band_weights(feature_extractor.w_f, feature_extractor.w_mel, feature_extractor.f, feature_extractor.m, plot_path=plot_path, name='signal_mfcc_weights', show_plot=show_plot)


def showcase_wavs(cfg, raw_plot=True, raw_energy_plot=True, spec_plot=True, mfcc_plot=True, use_mfcc_39=False, show_plot=False):
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
  wav_dir, anno_dir = '../docu/showcase_wavs/', '../docu/showcase_wavs/annotation/'

  # analyze some wavs
  for wav, anno in zip(glob(wav_dir + '*.wav'), glob(anno_dir + '*.TextGrid')):

    # info
    print("\nwav: ", wav), print("anno: ", anno)

    # load file
    x, _ = librosa.load(wav, sr=feature_params['fs'])

    # raw waveform
    if raw_plot: 
      plot_waveform(x, feature_params['fs'], fig_size=(8, 1), anno_file=anno, hop=feature_extractor.hop, plot_path='../docu/showcase_wavs/ignore/', name='signal_raw_showcase_' + wav.split('/')[-1].split('.')[0], axis_off=True, show_plot=show_plot)
      #plot_waveform(x, feature_params['fs'], anno_file=anno, hop=feature_extractor.hop, plot_path=plot_path, name='signal_raw_showcase_' + wav.split('/')[-1].split('.')[0], show_plot=show_plot)
    
    # raw energy plot
    if raw_energy_plot:

      # mfcc
      mfcc, bon_pos_mfcc = feature_extractor.extract_mfcc(x, reduce_to_best_onset=False)

      # energy frames
      e_win = np.sum(np.squeeze(view_as_windows(np.abs(x)**2, feature_extractor.raw_frame_size, step=1)), axis=1)
      e_win_mfcc = np.sum(np.squeeze(view_as_windows(mfcc[0, 0, :], feature_extractor.frame_size, step=1)), axis=1)

      # bon pos samples
      bon_pos = np.argmax(e_win)

      # plot
      plot_waveform(x, feature_params['fs'], e_samples=e_win, e_mfcc=e_win_mfcc, bon_mfcc=[bon_pos_mfcc, bon_pos_mfcc + feature_extractor.frame_size], bon_samples=[bon_pos, bon_pos + feature_extractor.raw_frame_size], y_ax_balance=False, anno_file=anno, hop=feature_extractor.hop, plot_path=plot_path, name='signal_onset_showcase_' + wav.split('/')[-1].split('.')[0], show_plot=show_plot)

    # spectrogram
    if spec_plot:
      plot_spec_profile(x, feature_extractor.calc_spectogram(x).T, feature_params['fs'], feature_extractor.N, feature_extractor.hop, anno_file=anno, plot_path=plot_path, name='signal_spec-lin_showcase_' + wav.split('/')[-1].split('.')[0], show_plot=show_plot)
      plot_spec_profile(x, feature_extractor.calc_spectogram(x).T, feature_params['fs'], feature_extractor.N, feature_extractor.hop, log_scale=True, anno_file=anno, plot_path=plot_path, name='signal_spec-log_showcase_' + wav.split('/')[-1].split('.')[0], show_plot=show_plot)

    # mfcc
    if mfcc_plot:
      mfcc, bon_pos = feature_extractor.extract_mfcc(x, reduce_to_best_onset=False)
      name = 'signal_mfcc_showcase_mfcc32_' + wav.split('/')[-1].split('.')[0] if not use_mfcc_39 else 'signal_mfcc_showcase_mfcc39_' + wav.split('/')[-1].split('.')[0]
      #plot_mfcc_plain(mfcc, plot_path='../docu/showcase_wavs/ignore/', name=name + '_plain', show_plot=show_plot)
      plot_mfcc_profile(x, cfg['feature_params']['fs'], feature_extractor.N, feature_extractor.hop, mfcc, anno_file=anno, sep_features=False, bon_pos=bon_pos, frame_size=cfg['feature_params']['frame_size'], plot_path=plot_path, name=name, close_plot=False, show_plot=show_plot)


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
  #if statistics_plot: audio_set1.analyze_dataset_extraction(calculate_overall_stats=True)
  if statistics_plot: audio_set1.analyze_dataset_extraction(plot_path=plot_path, file_pre='exp_dataset_', calculate_overall_stats=True, plot_hist=True, plot_damaged=False)

  # label table
  if label_table_plot: LatexTableMakerAudiosetLabels(audio_set1.all_label_file_dict, caption='all labels', label='tab:exp_dataset_all_labels', out_file=plot_path_tab + 'tab_exp_dataset_all_labels__not_released.tex')

  # plot wav grid
  if wav_grid_plot: 
    plot_wav_grid(audio_set1.extract_wav_examples(set_name='test', n_examples=1, from_selected_labels=False), feature_params=audio_set1.feature_params, grid_size=(6, 6), plot_path=plot_path, name='exp_dataset_wav_grid_speech_commands_v2', show_plot=True)
    plot_wav_grid(audio_set2.extract_wav_examples(set_name='my', n_examples=5), feature_params=audio_set2.feature_params, grid_size=(5, 5), plot_path=plot_path, name='exp_dataset_wav_grid_my', show_plot=True)


def batch_archive_grid_examples(cfg, show_plot=False):
  """
  batch archive examples from each label ploted as grid
  """

  # plot path
  plot_path = '../docu/thesis/5_exp/figs/'

  # audioset init
  audio_set1 = AudioDataset(cfg['datasets']['speech_commands'], feature_params=cfg['feature_params'], root_path='../')
  audio_set2 = AudioDataset(cfg['datasets']['my_recordings'], feature_params=cfg['feature_params'], root_path='../')

  # create batches
  batch_archive = SpeechCommandsBatchArchive(feature_file_dict={**audio_set1.feature_file_dict, **audio_set2.feature_file_dict}, batch_size_dict={'train': 32, 'test': 5, 'validation': 5, 'my': 1}, shuffle=False)

  # for all labels
  for l in ['left', 'right', 'up', 'down', 'go']:

    print("l: ", l)

    # create batches
    batch_archive.create_batches(selected_labels=[l])

    # plot
    plot_grid_images(batch_archive.x_batch_dict['train'][0, :30], context='mfcc', padding=1, num_cols=5, plot_path=plot_path, title='', name='exp_dataset_speech_cmd_mfcc_' + l, show_plot=show_plot)

  # create batches for my data
  batch_archive.create_batches()
  batch_archive.print_batch_infos()
  
  # plot my data
  plot_grid_images(np.squeeze(batch_archive.x_batch_dict['my'], axis=1), context='mfcc', padding=1, num_cols=5, plot_path=plot_path, title='', name='exp_dataset_my_mfcc', show_plot=show_plot)


def training_logs(cfg):
  """
  audio set wavs
  """

  # plot path
  plot_path_tab = '../docu/thesis/5_exp/tables/'
  log_path = '../docu/logs/'

  # cepstral table
  #LatexTableMakerCepstral(in_file=log_path + 'log_cepstral.log', out_file=plot_path_tab + 'tab_exp_fs_cepstral.tex', caption='Experiment on the impact of the amount of cepstral coefficient of MFCC features. Frame based normalization was evaluated additionally.', label='tab:exp_fs_cepstral')
  
  # cepstral table l12
  #LatexTableMakerCepstral(in_file=log_path + 'log_exp_cepstral_l12_2000.log', out_file=plot_path_tab + 'tab_exp_fs_cepstral_l12_2000.tex', caption='Experiment on the impact of the amount of cepstral coefficient of MFCC features with additional frame-based normalization evaluation, trained with 2000 epochs.', label='tab:exp_fs_cepstral_l12')
  
  # randomize frames
  #LatexTableMakerCepstral(in_file=log_path + 'log_exp_rand_frames_l12.log', out_file=plot_path_tab + 'tab_exp_fs_rand_frames_l12.tex', caption='Experiment of not randomizing frame positions.', label='tab:exp_fs_rand_frames_l12')

  # feature selection
  #LatexTableMakerMFCC(in_file=log_path + 'log_exp_mfcc_l12.log', out_file=plot_path_tab + 'tab_exp_fs_mfcc_l12.tex', caption='Experiment on the impact of feature enhancement of cepstral coefficients (c), deltas (d), double deltas (dd) and energy vectors (e).', label='tab:exp_fs_mfcc_l12')

  # adv label
  #LatexTableMakerAdv(in_file=log_path + 'log_exp_adv_label_l12.log', out_file=plot_path_tab + 'tab_exp_adv_label_l12.tex', caption='Experiment with adversarial label pre-training, using either Generator \enquote{g} or Discriminator \enquote{d} weights.', label='tab:exp_adv_label_l12')
  
  # adv dual
  #LatexTableMakerAdv(in_file=log_path + 'log_exp_adv_dual_l12.log', out_file=plot_path_tab + 'tab_exp_adv_dual_l12.tex', caption='Experiment with adversarial dual pre-training, using either Generator \enquote{g} or Discriminator \enquote{d} weights.', label='tab:exp_adv_dual_l12')
  
  # final
  LatexTableMakerFinal(in_file=log_path + 'log_exp_final_l12.log', out_file=plot_path_tab + 'tab_exp_final_l12.tex', caption='Experiment on whole dataset with 3500 examples per label, with 12 MFCC coefficients and frame-based normalization, trained with 2000 epochs.', label='tab:exp_final_l12')


def nn_theory():
  """
  neural network theory
  """

  # plot path
  plot_path = '../docu/thesis/4_nn/figs/'
  #plot_path = None

  # x-axis
  x = np.linspace(-10, 10, 100)

  # plot activation functions
  plot_activation_function(x, torch.sigmoid(torch.from_numpy(x)), plot_path=plot_path, name='nn_theory_activation_sigmoid', show_plot=True)
  plot_activation_function(x, torch.tanh(torch.from_numpy(x)), plot_path=plot_path, name='nn_theory_activation_tanh', show_plot=True)
  plot_activation_function(x, torch.relu(torch.from_numpy(x)), plot_path=plot_path, name='nn_theory_activation_relu', show_plot=True)


def noise_wavs_info(cfg):
  """
  get infos from provided noise wavs
  """

  # noise wav path
  wav_path = '../ignore/dataset/speech_commands_v0.02/_background_noise_/'

  # wav len dict
  wav_len_dict = {wav.split('/')[-1]: len(librosa.load(wav, sr=cfg['feature_params']['fs'])[0]) for wav in glob(wav_path + '*.wav')}

  # wav len to seconds
  wav_sec_dict = {k: v / cfg['feature_params']['fs'] for k, v in wav_len_dict.items()}

  # some prints
  print(wav_len_dict), print(wav_sec_dict)
  print("sum: {:.2f}s".format(sum(wav_sec_dict.values())))
  print("shift for 3500 examples: ", (sum(wav_sec_dict.values()) - 1) / 3500)


def test_bench_stuff(cfg):
  """
  test bench stuff
  """

  # create test bench
  test_bench = TestBench(cfg['test_bench'], test_model_path='../docu/best_models/ignore/exp_final/conv-jim/v5_c12n1m1_n-3500_r1-5_mfcc32-12_c1d0d0e0_norm1_f-1x12x50/bs-32_it-2000_lr-0p0001_adv-pre_bs-32_it-100_lr-d-0p0001_lr-g-0p0001_label_model-g/', root_path='../')

  # shift invariance test
  test_bench.test_invariances()



if __name__ == '__main__':
  """
  main
  """

  import yaml

  # yaml config file
  cfg = yaml.safe_load(open("../config.yaml"))

  # mfcc stuff
  #mfcc_stuff(cfg, dct_plot=True, mel_scale_plot=True, mel_band_plot=True, show_plot=True)

  # showcase wavs
  #showcase_wavs(cfg, raw_plot=True, raw_energy_plot=False, spec_plot=False, mfcc_plot=False, use_mfcc_39=False, show_plot=True)

  # feature selection tables
  #feature_selection_tables(overwrite=True)

  # audio set wavs
  #audio_set_wavs(cfg, statistics_plot=True, wav_grid_plot=False, label_table_plot=False)

  # batch archive
  #batch_archive_grid_examples(cfg, show_plot=True)

  # logs
  #training_logs(cfg)

  # theory
  #nn_theory()

  # noise wavs
  #noise_wavs_info(cfg)

  # test bench
  test_bench_stuff(cfg)

