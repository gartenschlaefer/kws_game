"""
mic class
"""

import numpy as np
import queue

# sound stuff
import sounddevice as sd

# my stuff
from feature_extraction import calc_mfcc39, onset_energy_level, find_min_energy_region
from collector import Collector
from classifier import Classifier


class Mic():
  """
  Mic class
  """

  def __init__(self, fs, N, hop, classifier, fs_device=48000, channels=1, energy_thres=5e-5, frame_size=32):

    # vars
    self.fs = fs
    self.N = N
    self.hop = hop
    self.classifier = classifier
    self.fs_device = fs_device
    self.channels = channels
    self.energy_thres = energy_thres
    self.frame_size = frame_size

    # queue and collector
    self.q = queue.Queue()
    self.collector = Collector(N=N, hop=hop, frame_size=32, update_size=32, frames_post=32)

    # determine downsample
    self.downsample = fs_device // fs

    # select microphone
    sd.default.device = 7

    # show devices
    print("device list: \n", sd.query_devices())

    # setup stream sounddevice
    self.stream = sd.InputStream(samplerate=self.fs*self.downsample, blocksize=self.hop*self.downsample, channels=channels, callback=self.callback_mic)

    # mic sleep caller
    sd.sleep(int(100))


  def callback_mic(self, indata, frames, time, status):
    """
    Input Stream Callback
    """

    if status:
      print(status)

    # put into queue
    self.q.put(indata[::self.downsample, 0])


  def read_mic_data(self):
    """
    reads the input from the queue
    """

    # init
    x = np.empty(shape=(0), dtype=np.float32)

    # onset flag
    is_onset = False

    # process data
    try:

      # get data
      x = self.q.get_nowait()

      # detect onset
      _, is_onset = onset_energy_level(x, alpha=self.energy_thres)

      # collection update
      self.collector.update_collect(x.copy())
      
    # no data
    except queue.Empty:
      return x, is_onset

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

      # mfcc
      mfcc = calc_mfcc39(x_onset, self.fs, N=self.N, hop=self.hop, n_filter_bands=32, n_ceps_coeff=12)

      # find best region
      _, bon_pos = find_min_energy_region(mfcc, self.fs, self.hop)

      # region of interest
      mfcc_bon = mfcc[:, bon_pos:bon_pos+self.frame_size]

      # classify collection
      return self.classifier.classify_sample(mfcc_bon)

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
  classifier = Classifier(file='./ignore/models/best_models/fstride_c-5.npz')  

  # create mic instance
  mic = Mic(fs=fs, N=N, hop=hop, classifier=classifier)

  # stream and update
  with mic.stream:

    print("recording...")
    while True:

      # get command
      command = mic.update_read_command()

      # interpret command
      if command is not None:

        print("yey command: ", command)
        # interpret command
        pass

