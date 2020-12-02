"""
mic class
"""

import numpy as np
import queue

# sound stuff
import sounddevice as sd

# my stuff
from feature_extraction import FeatureExtractor, onset_energy_level
from collector import Collector
from classifier import Classifier

from plots import plot_mfcc_profile


class Mic():
  """
  Mic class
  """

  def __init__(self, classifier, feature_params, mic_params, is_audio_record=False):

    # arguments
    self.classifier = classifier
    self.feature_params = feature_params
    self.mic_params = mic_params
    self.is_audio_record = is_audio_record

    # windowing params
    self.N, self.hop = int(feature_params['N_s'] * feature_params['fs']), int(feature_params['hop_s'] * feature_params['fs'])

    # queue
    self.q = queue.Queue()

    # collector
    self.collector = Collector(N=self.N, hop=self.hop, frame_size=self.feature_params['frame_size'], update_size=self.mic_params['update_size'], frames_post=self.mic_params['frames_post'], is_audio_record=self.is_audio_record)

    # feature extractor
    self.feature_extractor = FeatureExtractor(self.feature_params['fs'], N=self.N, hop=self.hop, n_filter_bands=self.feature_params['n_filter_bands'], n_ceps_coeff=self.feature_params['n_ceps_coeff'], frame_size=self.feature_params['frame_size'])

    # select microphone yourself (usually not necessary)
    if mic_params['select_device']:
      sd.default.device = self.mic_params['device']

    # show devices
    print("\ndevice list: \n", sd.query_devices())

    # setup stream sounddevice
    self.stream = sd.InputStream(samplerate=self.feature_params['fs'], blocksize=self.hop, channels=self.mic_params['channels'], callback=self.callback_mic)


  def callback_mic(self, indata, frames, time, status):
    """
    Input Stream Callback
    """

    if status:
      print(status)

    self.q.put(indata[:, 0].copy())


  def clear_mic_queue(self):
    """
    clear the queue after classification
    """

    # process data
    for i in range(self.q.qsize()):

      # get chunk
      x = self.q.get()

      # onset and energy archiv
      e, _ = onset_energy_level(x, alpha=self.mic_params['energy_thres'])

      # update collector
      self.collector.x_all = np.append(self.collector.x_all, x)
      self.collector.e_all = np.append(self.collector.e_all, e)


  def read_mic_data(self):
    """
    reads the input from the queue
    """

    # init
    x_collect = np.empty(shape=(0), dtype=np.float32)
    e_collect = np.empty(shape=(0), dtype=np.float32)

    # onset flag
    is_onset = False

    # process data
    if self.q.qsize():

      for i in range(self.q.qsize()):

        # get data
        x = self.q.get()

        # append chunk
        x_collect = np.append(x_collect, x.copy())

        # append energy level
        e_collect = np.append(e_collect, 1)

      # detect onset
      e_onset, is_onset = onset_energy_level(x_collect, alpha=self.mic_params['energy_thres'])

      # collection update
      self.collector.update_collect(x_collect.copy(), e=e_collect.copy()*e_onset, on=is_onset)

    return is_onset


  def update_read_command(self):
    """
    update mic
    """

    # read chunk
    is_onset = self.read_mic_data()

    # onset was detected
    if is_onset:

      # start collection of items
      self.collector.start_collecting()

    # collection is full
    if self.collector.is_full():

      # read out collection
      x_onset = self.collector.read_collection()

      # extract features
      mfcc_bon, bon_pos = self.feature_extractor.extract_mfcc39(x_onset)

      # classify collection
      y_hat, label = self.classifier.classify_sample(mfcc_bon)

      # plot
      plot_mfcc_profile(x_onset[bon_pos*self.hop:(bon_pos+self.feature_params['frame_size'])*self.hop], self.feature_params['fs'], self.N, self.hop, mfcc_bon, frame_size=self.feature_params['frame_size'], plot_path=self.mic_params['plot_path'], name='collect-{}_label-{}'.format(self.collector.collection_counter, label), enable_plot=self.mic_params['enable_plot'])

      # clear read queue
      self.clear_mic_queue()

      return y_hat

    return None


  def stop_mic_condition(self, time_duration):
    """
    stop mic if time duration is exceeded (memory issue in recording)
    """

    return (self.collector.x_all.shape[0] >= (time_duration * self.feature_params['fs'])) and self.is_audio_record


  def save_audio_file(self):
    """
    saves collection to audio file
    """

    # has not recorded audio
    if not self.is_audio_record:
      print("***you did not set the record flag!")
      return

    import soundfile

    # save audio
    soundfile.write('{}out_audio.wav'.format(self.mic_params['plot_path']), self.collector.x_all, self.feature_params['fs'], subtype=None, endian=None, format=None, closefd=True)


if __name__ == '__main__':
  """
  mic
  """

  import yaml
  import matplotlib.pyplot as plt
  import soundfile

  from plots import plot_waveform
  from common import create_folder

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # create folder
  create_folder([cfg['mic_params']['plot_path']])

  # window and hop size
  N, hop = int(cfg['feature_params']['N_s'] * cfg['feature_params']['fs']), int(cfg['feature_params']['hop_s'] * cfg['feature_params']['fs'])

  # classifier
  classifier = Classifier(model_path='./models/conv-fstride/v3_c-5_n-2000/bs-32_it-1000_lr-1e-05/', model_file_name='model.pth', params_file_name='params.npz', verbose=True)

  # create mic instance
  mic = Mic(classifier=classifier, feature_params=cfg['feature_params'], mic_params=cfg['mic_params'], is_audio_record=True)

  # stream and update
  with mic.stream:

    print("recording...")
    while not mic.stop_mic_condition(time_duration=2):

      # get command
      command = mic.update_read_command()

      # check if command
      if command is not None:

        # print command
        print("command: ", command)

    # clear queue
    mic.clear_mic_queue()

  # some prints
  print("x_all: ", mic.collector.x_all.shape)
  print("e_all: ", mic.collector.e_all.shape)
  print("on_all: ", mic.collector.on_all.shape)

  # plot waveform
  plot_waveform(mic.collector.x_all, cfg['feature_params']['fs'], mic.collector.e_all * 10, hop, mic.collector.on_all, title='input stream', ylim=(-1, 1), plot_path=None, name='None')

  # save audio
  mic.save_audio_file()

  # show plots
  plt.show()
