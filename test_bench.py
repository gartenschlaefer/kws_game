"""
Test Bench for quality investigation of neural network models
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import re

# my stuff
from common import create_folder
from feature_extraction import FeatureExtractor, frames_to_sample
from net_handler import NetHandler
from plots import plot_waveform, plot_test_bench_shift, plot_test_bench_noise
from legacy import legacy_adjustments_net_params

# other
from skimage.util.shape import view_as_windows
from glob import glob


class TestBench():
  """
  test bench class for evaluating models
  """

  def __init__(self, cfg_tb, test_model_path, root_path='./'):

    # arguments
    self.cfg_tb = cfg_tb
    self.test_model_path = test_model_path
    self.root_path = root_path

    # shortcuts
    self.feature_params, self.data_size = None, None

    # paths
    self.paths = dict((k, self.root_path + v) for k, v in self.cfg_tb['paths'].items())

    # test model path
    self.test_model_name = self.test_model_path.split('/')[-2]

    # determine available model files
    model_files_av = [f.split('/')[-1] for f in glob(self.test_model_path + '*model.pth')]

    # model file
    self.model_files = [self.test_model_path + f for f in model_files_av if f in self.cfg_tb['model_file_names']]

    # pick just the first one (errors should not occur)
    self.model_file = self.model_files[0]

    # param file
    self.params_file = self.test_model_path + self.cfg_tb['params_file_name']

    # wavs
    self.test_wavs = [self.root_path + wav for wav in self.cfg_tb['test_wavs']]

    # create folder
    create_folder(list(self.paths.values()))

    # parameter loading
    net_params = np.load(self.params_file, allow_pickle=True)

    # extract params
    self.nn_arch, self.train_params, self.class_dict = net_params['nn_arch'][()], net_params['train_params'][()], net_params['class_dict'][()]

    # legacy stuff
    #self.data_size, self.feature_params = self.legacy_adjustments_tb(net_params)

    # legacy stuff
    self.data_size, self.feature_params = legacy_adjustments_net_params(net_params)

    # init feature extractor
    self.feature_extractor = FeatureExtractor(self.feature_params)

    # init net handler
    self.net_handler = NetHandler(nn_arch=self.nn_arch, class_dict=self.class_dict, data_size=self.data_size, use_cpu=True)

    # load model
    self.net_handler.load_models(model_files=[self.model_file])

    # set evaluation mode
    self.net_handler.set_eval_mode()


  def test_invariances(self):
    """
    test all invariances
    """

    # init lists
    all_labels, all_corrects_shift, all_corrects_noise, all_probs_shift, all_probs_noise = [], [], [], [], []

    # test model
    print("\n--Test Bench\ntest model: [{}]".format(self.test_model_name))

    # go through each test wav
    for wav in self.test_wavs:

      # print message
      print("\ntest wav: ", wav)

      # file naming extraction
      file_name, file_index, actual_label = self.file_naming_extraction(wav)

      # update labels
      all_labels.append(actual_label)

      # read audio from file
      x_wav, _ = librosa.load(wav, sr=self.feature_params['fs'])

      # shift invariance
      corrects_shift, probs_shift = self.test_shift_invariance(x_wav, actual_label)

      # noise invariance
      corrects_noise, probs_noise = self.test_noise_invariance(x_wav, actual_label, mu=0)

      # collect corrects
      all_corrects_shift.append(corrects_shift)
      all_corrects_noise.append(corrects_noise)

      all_probs_shift.append(probs_shift)
      all_probs_noise.append(probs_noise)

    # some prints
    if self.cfg_tb['enable_info_prints']: print("\nall_corrects_shift:\n", all_corrects_shift), print("\nall_corrects_noise:\n", all_corrects_noise), print("\nall labels: ", all_labels)

    # plots
    plot_test_bench_shift(x=all_corrects_shift, y=all_labels, context='bench-shift-2', title='shift ' + self.test_model_name, plot_path=self.test_model_path, name='test_bench_shift', show_plot=False)
    plot_test_bench_shift(x=all_probs_shift, y=all_labels, context='bench-shift', title='shift ' + self.test_model_name, plot_path=self.test_model_path, name='test_bench_shift-prob', show_plot=False)

    plot_test_bench_noise(x=all_corrects_noise, y=all_labels, snrs=self.cfg_tb['snrs'], context='bench-noise-2', title='noise ' + self.test_model_name, plot_path=self.test_model_path, name='test_bench_noise', show_plot=False)
    plot_test_bench_noise(x=all_probs_noise, y=all_labels, snrs=self.cfg_tb['snrs'], context='bench-noise', title='noise ' + self.test_model_name, plot_path=self.test_model_path, name='test_bench_noise-prob', show_plot=False)


  def test_noise_invariance(self, x_wav, actual_label, mu=0):
    """
    test model against noise invariance
    """

    # init lists
    pred_label_list, probs = [], []

    # origin
    if self.cfg_tb['enable_plot']: plot_waveform(x_wav, self.feature_params['fs'], title='origin actual: [{}]'.format(actual_label), plot_path=self.paths['shift_wavs'], name='{}_origin'.format(actual_label))

    # test model with different snr values
    for snr in self.cfg_tb['snrs']:

      # signal power
      p_x_eff = x_wav @ x_wav.T / len(x_wav)

      # calculate noise signal power
      sigma = np.sqrt(p_x_eff / (10**(snr / 10)))

      # noise generation
      n = np.random.normal(mu, sigma, len(x_wav))

      # add noise
      x_noise = x_wav + n

      # noise signal power
      p_n_eff = n @ n.T / len(n)

      # print energy info
      # print("sigma: ", sigma), print("p_x: ", p_x_eff), print("p_n: ", p_n_eff), print("db: ", 10 * np.log10(p_x_eff / p_n_eff))

      # feature extraction
      #x_mfcc, _ = self.feature_extractor.extract_mfcc(x_noise, reduce_to_best_onset=True)

      # feature extraction
      x, _ = self.feature_extractor.extract_mfcc(x_noise, reduce_to_best_onset=True) if self.net_handler.nn_arch != 'wavenet' else self.feature_extractor.get_best_raw_samples(x_noise, add_channel_dim=True)

      # classify
      y_hat, o, pred_label = self.net_handler.classify_sample(x)

      # append predicted label and probs
      pred_label_list.append(pred_label)
      probs.append(float(o[0, self.class_dict[actual_label]]))

      # plot wavs
      if self.cfg_tb['enable_plot']: plot_waveform(x_noise, self.feature_params['fs'], title='snr: [{}] actual: [{}] pred: [{}]'.format(snr, actual_label, pred_label), plot_path=self.paths['noise_wavs'], name='{}_snr{}'.format(actual_label, snr))

    # correct list
    corrects = [int(actual_label == l) for l in pred_label_list]

    # print message
    print("test bench noise acc: ", np.sum(corrects) / len(corrects))

    return corrects, probs


  def test_shift_invariance(self, x_wav, actual_label):
    """
    test model against shift invariance
    """

    # init lists
    pred_label_list, probs = [], []

    # feature extraction
    x, _ = self.feature_extractor.extract_mfcc(x_wav, reduce_to_best_onset=False) if self.net_handler.nn_arch != 'wavenet' else (x_wav[np.newaxis, :], 0)

    # windowed
    x_win = np.squeeze(view_as_windows(x, self.data_size, step=self.cfg_tb['shift_frame_step']), axis=(0, 1)) if self.net_handler.nn_arch != 'wavenet' else np.squeeze(view_as_windows(x, self.data_size, step=self.cfg_tb['shift_frame_step'] * self.feature_extractor.hop), axis=0)

    for i, x in enumerate(x_win):

      # classify
      y_hat, o, pred_label = self.net_handler.classify_sample(x)

      # append predicted label
      pred_label_list.append(pred_label)
      probs.append(float(o[0, self.class_dict[actual_label]]))

      # plot
      time_s = frames_to_sample(i * self.cfg_tb['shift_frame_step'], self.feature_params['fs'], self.feature_extractor.hop)
      time_e = frames_to_sample(i * self.cfg_tb['shift_frame_step'] + self.feature_params['frame_size'], self.feature_params['fs'], self.feature_extractor.hop)

      # plot waveform
      if self.cfg_tb['enable_plot']: plot_waveform(x_wav[time_s:time_e], self.feature_params['fs'], title='frame{} actual: [{}] pred: [{}]'.format(i, actual_label, pred_label), plot_path=self.paths['shift_wavs'], name='{}_frame{}'.format(actual_label, i))

    # correct list
    corrects = [int(actual_label == l) for l in pred_label_list]

    # print message
    print("test bench shift acc: ", np.sum(corrects) / len(corrects))

    return corrects, probs


  def file_naming_extraction(self, file, file_ext='.wav'):
    """
    extract file name ergo label
    """

    # extract filename
    file_name = re.findall(r'[\w+ 0-9]*' + re.escape(file_ext), file)[0]

    # extract file index from filename
    file_index = re.sub(r'[a-z A-Z]|(' + re.escape(file_ext) + r')', '', file_name)

    # extract label from filename
    label = re.sub(r'([0-9]*' + re.escape(file_ext) + r')', '', file_name)

    return file_name, file_index, label



if __name__ == '__main__':
  """
  main
  """
  
  import yaml
  
  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # create test bench
  test_bench = TestBench(cfg['test_bench'], test_model_path='./ignore/test_bench/test_models/conv-encoder/')

  # shift invariance test
  test_bench.test_invariances()








