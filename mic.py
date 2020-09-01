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

#from plots import plot_mfcc_profile


class Mic():
  """
  Mic class
  """

  def __init__(self, fs, N, hop, classifier, fs_device=48000, channels=1, energy_thres=1e-4, frame_size=32, device=7, is_audio_record=False):

    # vars
    self.fs = fs
    self.N = N
    self.hop = hop
    self.classifier = classifier
    self.fs_device = fs_device
    self.channels = channels
    self.energy_thres = energy_thres
    self.frame_size = frame_size
    self.device = device

    # queue and collector
    self.q = queue.Queue()
    self.collector = Collector(N=N, hop=hop, frame_size=32, update_size=32, frames_post=32, is_audio_record=is_audio_record)

    # feature extractor
    self.feature_extractor = FeatureExtractor(self.fs, N=self.N, hop=self.hop, n_filter_bands=32, n_ceps_coeff=12, frame_size=self.frame_size)

    # determine downsample
    self.downsample = fs_device // fs

    # select microphone
    sd.default.device = self.device

    # show devices
    print("\ndevice list: \n", sd.query_devices())

    # setup stream sounddevice
    self.stream = sd.InputStream(samplerate=self.fs*self.downsample, blocksize=self.hop*self.downsample, channels=channels, callback=self.callback_mic)


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

    # onset flag
    is_onset = False

    # process data
    if self.q.qsize():

      # read out data
      while not self.q.empty():

        # get data
        x = self.q.get_nowait()

        # concatenate for energy level
        x_collect = np.concatenate((x_collect, x))

        # collection update
        self.collector.update_collect(x.copy())

      # detect onset
      _, is_onset = onset_energy_level(x_collect, alpha=self.energy_thres)

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
      y_hat = self.classifier.classify_sample(mfcc_bon)

      #plot_mfcc_profile(x_onset[bon_pos*self.hop:(bon_pos+32)*self.hop], self.fs, self.N, self.hop, mfcc_bon, frame_size=32, plot_path='./delete/', name='collect-{}'.format(self.collector.collection_counter))

      # clear read queue
      self.clear_mic_queue()

      return y_hat

    return None


if __name__ == '__main__':
  """
  mic
  """

  # params
  fs = 16000

  # window and hop size
  N, hop = int(0.025 * fs), int(0.010 * fs)

  # create classifier
  classifier = Classifier(file='./models/fstride_c-5.npz', verbose=True) 

  # create mic instance
  mic = Mic(fs=fs, N=N, hop=hop, classifier=classifier)

  # stream and update
  with mic.stream:

    print("recording...")
    while True:

      # get command
      command = mic.update_read_command()

