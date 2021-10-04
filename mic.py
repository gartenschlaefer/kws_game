"""
mic class
"""

import numpy as np
import queue
import yaml
import os

# sound stuff
import sounddevice as sd
import soundfile
import contextlib

# my stuff
from feature_extraction import FeatureExtractor
from collector import Collector
from classifier import Classifier
from common import create_folder
from plots import plot_mfcc_profile


class Mic():
  """
  Mic class
  """

  def __init__(self, classifier, mic_params, is_audio_record=False, root_path='./'):

    # arguments
    self.classifier = classifier
    self.mic_params = mic_params
    self.is_audio_record = is_audio_record
    self.root_path = root_path

    # user settings
    self.user_settings_file = self.root_path + self.mic_params['user_settings_file']

    # plot path
    self.plot_path = self.root_path + self.mic_params['plot_path']

    # create folder for plot path
    create_folder([self.plot_path])

    # shortcuts
    self.feature_params = classifier.feature_params

    # feature extractor
    self.feature_extractor = FeatureExtractor(self.feature_params)

    # windowing params
    self.N, self.hop = self.feature_extractor.N, self.feature_extractor.hop

    # queue
    self.q = queue.Queue()

    # collector
    self.collector = Collector(N=self.N, hop=self.hop, frame_size=self.feature_params['frame_size'], update_size=self.mic_params['update_size'], frames_post=self.mic_params['frames_post'], is_audio_record=self.is_audio_record)

    # device
    self.device = sd.default.device[0] if not self.mic_params['select_device'] else self.mic_params['device']

    # determine downsample
    self.downsample = self.mic_params['fs_device'] // self.feature_params['fs']

    # get input devices
    self.input_dev_dict = self.extract_devices()

    # show devices
    print("\ndevice list: \n", sd.query_devices())
    print("\ninput device ids: ", self.input_dev_dict.keys())

    # energy threshold in lin scale
    self.energy_thresh = 10**(self.mic_params['energy_thresh_db'] / 10)

    # stream
    self.stream = contextlib.nullcontext()

    # change device flag
    self.change_device_flag = False

    # steam active
    self.stream_active = False


  def load_user_settings(self, user_settings_file):
    """
    load user settings like device and energy threshold
    """

    # load user settings
    user_settings = yaml.safe_load(open(user_settings_file)) if os.path.isfile(user_settings_file) else {}

    # update mic params
    self.mic_params.update(user_settings)

    # device
    self.device = sd.default.device[0] if not self.mic_params['select_device'] else self.mic_params['device']

    # energy threshold in lin scale
    self.energy_thresh = 10**(self.mic_params['energy_thresh_db'] / 10)


  def init_stream(self, enable_stream=True, load_user_settings_file=True):
    """
    init stream
    """

    # load user settings
    if load_user_settings_file: self.load_user_settings(self.user_settings_file)

    # init stream
    self.stream = sd.InputStream(device=self.device, samplerate=self.mic_params['fs_device'], blocksize=int(self.hop * self.downsample), channels=self.mic_params['channels'], callback=self.callback_mic) if enable_stream else contextlib.nullcontext()
    
    # flags
    self.change_device_flag = False
    self.stream_active = True if enable_stream else False


  def change_device(self, device):
    """
    change to device
    """
    self.change_device_flag = True
    self.device = device
    print("changed device: ", device)


  def change_energy_thresh_db(self, e):
    """
    change energy threshold
    """
    self.mic_params['energy_thresh_db'] = e
    self.energy_thresh = 10**(e / 10)
    print("changed energy thresh: ", e)


  def extract_devices(self):
    """
    extract only input devices
    """
    return {i:dev for i, dev in enumerate(sd.query_devices()) if dev['max_input_channels']}


  def callback_mic(self, indata, frames, time, status):
    """
    Input Stream Callback
    """

    # debug
    if status: print(status)

    #self.q.put(indata[:, 0].copy())

    # add to queue with primitive downsampling
    self.q.put(indata[::self.downsample, 0].copy())


  def clear_mic_queue(self):
    """
    clear the queue after classification
    """

    # process data
    for i in range(self.q.qsize()):

      # get chunk
      x = self.q.get()

      # onset and energy archiv
      e, _ = self.onset_energy_level(x, alpha=self.energy_thresh)

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
      e_onset, is_onset = self.onset_energy_level(x_collect, alpha=self.energy_thresh)

      # collection update
      self.collector.update_collect(x_collect.copy(), e=e_collect.copy()*e_onset, on=is_onset)

    return is_onset


  def onset_energy_level(self, x, alpha=0.01):
    """
    onset detection with energy level, x: [n]
    """

    # energy calculation
    e = x.T @ x / len(x)

    return e, e > alpha


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
      x_feature_bon, bon_pos = self.feature_extractor.extract_audio_features(x_onset)

      # classify collection
      y_hat, label = self.classifier.classify(x_feature_bon)

      # plot
      if self.mic_params['enable_plot']: plot_mfcc_profile(x_onset[bon_pos*self.hop:(bon_pos+self.feature_params['frame_size'])*self.hop], self.feature_params['fs'], self.N, self.hop, x_feature_bon, frame_size=self.feature_params['frame_size'], plot_path=self.plot_path, name='collect-{}_label-{}'.format(self.collector.collection_counter, label))

      # clear read queue
      self.clear_mic_queue()

      return label

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

    # save audio
    soundfile.write('{}out_audio.wav'.format(self.plot_path), self.collector.x_all, self.feature_params['fs'], subtype=None, endian=None, format=None, closefd=True)


if __name__ == '__main__':
  """
  mic
  """

  from plots import plot_waveform

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # classifier
  classifier = Classifier(cfg_classifier=cfg['classifier'])
  
  # create mic instance
  mic = Mic(classifier=classifier, mic_params=cfg['mic_params'], is_audio_record=True)

  # init stream
  mic.init_stream()
  
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
  plot_waveform(mic.collector.x_all, cfg['feature_params']['fs'], e=mic.collector.e_all * 10, hop=mic.hop, onset_frames=mic.collector.on_all, title='input stream', ylim=(-1, 1), plot_path=None, name='None', show_plot=True)

  # save audio
  mic.save_audio_file()
