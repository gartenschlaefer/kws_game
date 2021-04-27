"""
thesis visualizations
"""

import numpy as np
import librosa
import re
from glob import glob

import sys
sys.path.append("../")

from audio_dataset import AudioDataset
from feature_extraction import FeatureExtractor, custom_dct_matrix
from plots import plot_mel_band_weights, plot_mfcc_profile, plot_waveform, plot_dct, plot_wav_grid, plot_spectogram
from latex_table_maker import LatexTableMaker


def mfcc_stuff(cfg):
  """
  for dct, filter bands, etc
  """

  # plot path
  plot_path = '../docu/thesis/3_theory/figs/a3_mfcc/'

  # init feature extractor
  feature_extractor = FeatureExtractor(cfg['feature_params'])

  # plot dct
  plot_dct(custom_dct_matrix(cfg['feature_params']['n_filter_bands']), plot_path=plot_path, name='dct', show_plot=False)
  plot_dct(custom_dct_matrix(cfg['feature_params']['n_filter_bands']), plot_path=plot_path, context='dct-div', name='dct-div', show_plot=False)

  # plot mel bands
  plot_mel_band_weights(feature_extractor.w_f, feature_extractor.w_mel, feature_extractor.f, feature_extractor.m, plot_path=plot_path, name='weights', show_plot=True)


def showcase_wavs(cfg, raw_plot=True, spec_plot=True):
  """
  showcase wavs
  """

  # plot path
  plot_path_raw = '../docu/thesis/3_theory/figs/a1_raw/'
  plot_path_spec = '../docu/thesis/3_theory/figs/a2_spectogram/'

  # init feature extractor
  feature_extractor = FeatureExtractor(cfg['feature_params'])

  # wav, anno dir
  wav_dir, anno_dir = '../ignore/my_recordings/showcase_wavs/', '../ignore/my_recordings/showcase_wavs/annotation/'

  # analyze some wavs
  for wav, anno in zip(glob(wav_dir + '*.wav'), glob(anno_dir + '*.TextGrid')):

    # info
    print("\nwav: ", wav), print("anno: ", anno)

    # load file
    x, _ = librosa.load(wav, sr=cfg['feature_params']['fs'])

    # mfcc extraction
    #mfcc, bon_pos = feature_extractor.extract_mfcc(x, reduce_to_best_onset=False)
    #plot_mfcc_profile(x, cfg['feature_params']['fs'], feature_extractor.N, feature_extractor.hop, mfcc, anno_file=anno, sep_features=True, diff_plot=False, bon_pos=bon_pos, frame_size=cfg['feature_params']['frame_size'], plot_path=wav_dir, name=wav.split('/')[-1].split('.')[0], show_plot=True)
    
    # raw waveform
    if raw_plot: plot_waveform(x, cfg['feature_params']['fs'], anno_file=anno, hop=feature_extractor.hop, context='wav', title=wav.split('/')[-1].split('.')[0]+'_my', plot_path=plot_path_raw, name='raw_' + wav.split('/')[-1].split('.')[0] + '_my', show_plot=True)
    
    if spec_plot:
      x_spec = feature_extractor.calc_spectogram(x)
      plot_spectogram(x, x_spec.T, plot_path=plot_path_spec, name='spec', show_plot=True)

    # plt.figure()
    # librosa.display.specshow(x_spec.T, sr=16000, hop_length=160, x_axis='time')

    # plt.figure()
    # plt.imshow(x_spec.T)
    # plt.show()


def feature_selection_tables():
  """
  feature selection tables
  """

  # c30 it-1000
  #in_file, out_file = '../ignore/logs/ml_it1000_c30_features_fc1.log', '../docu/thesis/4_practice/tables/b1_feature_selection/ml_it1000_c30_features_fc1.tex'
  
  # c30 it-1000
  in_file, out_file = '../ignore/logs/ml_it2000_c30_features_fc3.log', '../docu/thesis/4_practice/tables/b1_feature_selection/ml_it2000_c30_features_fc3.tex'

  # table info
  print("feature selection table: ", out_file)

  # instances
  lt_maker = LatexTableMaker(in_file=in_file, extraction_type='feature_selection')

  # extract table
  tables = lt_maker.extract_table(out_file=out_file)


def audio_set_wavs(cfg):
  """
  audio set wavs
  """

  # plot path
  plot_path = '../docu/thesis/4_practice/figs/a_dataset/'

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
  showcase_wavs(cfg, raw_plot=False, spec_plot=True)

  # feature selection tables
  #feature_selection_tables()

  # audio set wavs
  #audio_set_wavs(cfg)


