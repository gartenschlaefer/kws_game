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

    # queue and collector
    self.q = queue.Queue()
    self.collector = Collector(N=self.N, hop=self.hop, frame_size=self.feature_params['frame_size'], update_size=self.mic_params['update_size'], frames_post=self.mic_params['frames_post'], is_audio_record=self.is_audio_record)

    # feature extractor
    self.feature_extractor = FeatureExtractor(self.feature_params['fs'], N=self.N, hop=self.hop, n_filter_bands=self.feature_params['n_filter_bands'], n_ceps_coeff=self.feature_params['n_ceps_coeff'], frame_size=self.feature_params['frame_size'])

    # determine downsample
    self.downsample = mic_params['fs_device'] // self.feature_params['fs']

    # select microphone
    sd.default.device = self.mic_params['device']

    # show devices
    print("\ndevice list: \n", sd.query_devices())

    # setup stream sounddevice
    self.stream = sd.InputStream(samplerate=self.feature_params['fs']*self.downsample, blocksize=self.hop*self.downsample, channels=self.mic_params['channels'], callback=self.callback_mic)


  def callback_mic(self, indata, frames, time, status):
    """
    Input Stream Callback
    """

    if status:
      print(status)

    # put into queue
    self.q.put(indata[::self.downsample, 0])


  def clear_mic_queue(self):
    """
    clear the queue after classification
    """

    # empty queue
    while not self.q.empty():
      dummy = self.q.get_nowait()


  def read_mic_data(self):
    """
    reads the input from the queue
    """

    # init
    x = np.empty(shape=(0), dtype=np.float32)
    x_collect = np.empty(shape=(0), dtype=np.float32)
    e_collect = np.empty(shape=(0), dtype=np.float32)

    # onset flag
    is_onset = False

    # process data
    if self.q.qsize():

      # read out data
      while not self.q.empty():

        # get data
        x = self.q.get_nowait()

        # concatenate collection
        x_collect = np.concatenate((x_collect, x))

        # append energy level
        e_collect = np.append(e_collect, 1)

      # detect onset
      e_onset, is_onset = onset_energy_level(x_collect, alpha=self.mic_params['energy_thres'])

      # collection update
      self.collector.update_collect(x_collect.copy(), e=e_collect.copy()*e_onset, on=is_onset)

    return x, is_onset


  def update_read_command(self):
    """
    update mic
    """

    # read chunk
    xi, is_onset = self.read_mic_data()

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
      plot_mfcc_profile(x_onset[bon_pos*self.hop:(bon_pos+32)*self.hop], self.feature_params['fs'], self.N, self.hop, mfcc_bon, frame_size=self.feature_params['frame_size'], plot_path=self.mic_params['plot_path'], name='collect-{}_label-{}'.format(self.collector.collection_counter, label), enable_plot=self.mic_params['enable_plot'])

      # clear read queue
      self.clear_mic_queue()

      return y_hat

    return None


  def stop_mic_condition(self, time_duration):
    """
    stop mic if time duration is exceeded (memory issue in recording)
    """

    return (self.collector.x_all.shape[0] >= (time_duration * self.feature_params['fs'])) and self.is_audio_record


if __name__ == '__main__':
  """
  mic
  """

  import yaml
  import matplotlib.pyplot as plt

  from plots import plot_waveform

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

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

  # some prints
  print("x_all: ", mic.collector.x_all.shape)
  print("e_all: ", mic.collector.e_all.shape)
  print("on_all: ", mic.collector.on_all.shape)

  # plot waveform
  plot_waveform(mic.collector.x_all, cfg['feature_params']['fs'], mic.collector.e_all * 10, hop, mic.collector.on_all, title='input stream', ylim=(-1, 1), plot_path=None, name='None')
  plt.show()

