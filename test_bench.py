"""
Test bench for investigating models qualities
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

# other
from skimage.util.shape import view_as_windows


class TestBench():
  """
  test bench class for evaluating models
  """

  def __init__(self, cfg_test_bench, feature_extractor=None, net_handler=None, class_dict=None, root_path='./'):

    # arguments
    self.cfg_test_bench = cfg_test_bench
    self.feature_extractor = feature_extractor
    self.net_handler = net_handler
    self.class_dict = class_dict
    self.root_path = root_path

    # shortcuts
    self.feature_params = None
    self.data_size = None

    # noise levels [db]
    #self.noise_levels = [0.0001, 0.001, 0.01, 0.1, 1, 2, 4]
    #self.snrs = [10, 6, 3, 0, -3, -6, -10]
    self.snrs = [16, 13, 10, 6, 3, 0, -3, -6, -10, -13, -16]

    # shift window step
    self.window_step = 1

    # paths
    self.paths = dict((k, self.root_path + v) for k, v in self.cfg_test_bench['paths'].items())
    self.plot_paths = dict((k, self.root_path + v) for k, v in self.cfg_test_bench['plot_paths'].items())

    # test model path
    self.test_model_path = self.root_path + self.cfg_test_bench['test_model_path']
    self.test_model_name = self.cfg_test_bench['test_model_path'].split('/')[-2]

    # model file
    self.model_file = self.test_model_path + self.cfg_test_bench['model_file_name']
    self.params_file = self.test_model_path + self.cfg_test_bench['params_file_name']

    # wavs
    self.test_wavs = [self.root_path + wav for wav in cfg['test_bench']['test_wavs']]

    # create folder
    create_folder(list(self.paths.values()) + list(self.plot_paths.values()))


  def load_net_handler_from_file(self, model_file, params_file):
    """
    net handler from file
    """

    self.model_file = model_file
    self.params_file = params_file

    # parameter loading
    net_params = np.load(params_file, allow_pickle=True)

    # class dictionary
    self.class_dict = net_params['class_dict'][()]

    # for legacy models
    try:
      self.data_size = data['data_size'][()]
    except:
      self.data_size = (1, 39, 32)
      print("old classifier model use fixed data size: {}".format(self.data_size))

    try:
      self.feature_params = data['feature_params'][()]
    except:
      self.feature_params = {'fs': 16000, 'N_s': 0.025, 'hop_s': 0.010, 'n_filter_bands': 32, 'n_ceps_coeff': 12, 'frame_size': 32, 'norm_features': False, 'use_channels': False, 'use_cepstral_features': True, 'use_delta_features': True, 'use_double_delta_features': True, 'use_energy_features': True}
      print("old classifier model use fixed feature parameters: {}".format(self.feature_params))

    # init feature extractor
    self.feature_extractor = FeatureExtractor(self.feature_params)

    # init net handler
    self.net_handler = NetHandler(nn_arch=net_params['nn_arch'][()], class_dict=self.class_dict, data_size=self.data_size, use_cpu=True)

    # load model
    self.net_handler.load_models(model_files=[self.model_file])

    # set evaluation mode
    self.net_handler.set_eval_mode()


  def test_invariances(self):
    """
    test all invariances
    """

    # check net handler
    if self.net_handler is None:
      print("*** net handler not specified!")
      return

    # init lists
    all_labels, all_corrects_shift, all_corrects_noise = [], [], []
    all_probs_shift, all_probs_noise = [], []

    # test model
    print("test model: ", self.test_model_name)

    # go through each test wav
    for wav in self.test_wavs:

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

    print("\nall_corrects_shift:\n", all_corrects_shift)
    print("\nall_corrects_noise:\n", all_corrects_noise)
    print("\nall labels: ", all_labels)

    # plots
    plot_test_bench_shift(x=all_corrects_shift, y=all_labels, title='shift ' + self.test_model_name, plot_path=self.plot_paths['shift'], name=self.test_model_name + '_shift', show_plot=False)
    plot_test_bench_shift(x=all_probs_shift, y=all_labels, title='shift ' + self.test_model_name, plot_path=self.plot_paths['shift'], name=self.test_model_name + '_shift-prob', show_plot=False)

    plot_test_bench_noise(x=all_corrects_noise, y=all_labels, snrs=self.snrs, title='noise ' + self.test_model_name, plot_path=self.plot_paths['noise'], name=self.test_model_name + '_noise', show_plot=False)
    plot_test_bench_noise(x=all_probs_noise, y=all_labels, snrs=self.snrs, title='noise ' + self.test_model_name, plot_path=self.plot_paths['noise'], name=self.test_model_name + '_noise-prob', show_plot=False)


  def test_noise_invariance(self, x_wav, actual_label, mu=0):
    """
    test model against noise invariance
    """

    # init lists
    pred_label_list, probs = [], []

    # origin
    #plot_waveform(x_wav, self.feature_params['fs'], title='origin actual: [{}]'.format(actual_label), plot_path=self.plot_paths['noise_wavs'], name='{}_origin'.format(actual_label))


    for snr in self.snrs:

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

      # print("sigma: ", sigma)
      # print("p_x: ", p_x_eff)
      # print("p_n: ", p_n_eff)
      # print("db: ", 10 * np.log10(p_x_eff / p_n_eff))

      # feature extraction
      x_mfcc, _ = self.feature_extractor.extract_mfcc(x_noise, reduce_to_best_onset=True)

      # classify
      y_hat, o, pred_label = self.net_handler.classify_sample(x_mfcc)

      # append predicted label and probs
      pred_label_list.append(pred_label)
      probs.append(float(o[0, self.class_dict[actual_label]]))

      # plot wavs
      #plot_waveform(x_noise, self.feature_params['fs'], title='snr: [{}] actual: [{}] pred: [{}]'.format(snr, actual_label, pred_label), plot_path=self.plot_paths['noise_wavs'], name='{}_snr{}'.format(actual_label, snr))

    # correct list
    corrects = [int(actual_label == l) for l in pred_label_list]

    print("noise acc: ", np.sum(corrects) / len(corrects))

    return corrects, probs


  def test_shift_invariance(self, x_wav, actual_label):
    """
    test model against shift invariance
    """

    # init lists
    pred_label_list, probs = [], []

    # feature extraction
    x_mfcc, _ = self.feature_extractor.extract_mfcc(x_wav, reduce_to_best_onset=False)

    # windowed
    x_win = np.squeeze(view_as_windows(x_mfcc, self.data_size, step=self.window_step), axis=(0, 1))

    for i, x in enumerate(x_win):

      # classify
      y_hat, o, pred_label = self.net_handler.classify_sample(x)

      # append predicted label
      pred_label_list.append(pred_label)
      probs.append(float(o[0, self.class_dict[actual_label]]))

      # plot
      time_s = frames_to_sample(i * self.window_step, self.feature_params['fs'], self.feature_extractor.hop)
      time_e = frames_to_sample(i * self.window_step + self.data_size[2], self.feature_params['fs'], self.feature_extractor.hop)

      # plot waveform
      #plot_waveform(x_wav[time_s:time_e], self.feature_params['fs'], title='frame{} actual: [{}] pred: [{}]'.format(i, actual_label, pred_label), plot_path=self.plot_paths['shift_wavs'], name='{}_frame{}'.format(actual_label, i))

    # correct list
    corrects = [int(actual_label == l) for l in pred_label_list]

    #print("shift corrects: ", corrects)
    print("shift acc: ", np.sum(corrects) / len(corrects))

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
  Insight file - gives insight in neural nets
  """
  
  import yaml
  
  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))



  # create test bench
  test_bench = TestBench(cfg['test_bench'])

  # load net handler from configs
  test_bench.load_net_handler_from_file(test_bench.model_file, test_bench.params_file)

  # shift invariance test
  test_bench.test_invariances()








