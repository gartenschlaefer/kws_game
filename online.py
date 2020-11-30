"""
Online Microphone input classification
"""

import numpy as np
import matplotlib.pyplot as plt

import queue
import time

# sound stuff
import sounddevice as sd

# my stuff
from common import *
from plots import *

from feature_extraction import onset_energy_level, FeatureExtractor
from collector import Collector
from mic import Mic
from classifier import Classifier


def callback_mic(indata, frames, time, status):
  """
  Input Stream Callback
  """

  if status:
    print(status)

  # put into queue
  q.put(indata[::downsample, 0])


def read_mic_data(collector):
  """
  reads the input from the queue
  """

  # init
  x = np.empty(shape=(0), dtype=np.float32)
  e = np.empty(shape=(0), dtype=np.float32)

  # onset flag
  is_onset = False

  # process data
  try:

    # get data
    x = q.get_nowait()

    # detect onset - mean energy: 0.0002405075
    #e_onset, is_onset = onset_energy_level(x, alpha=0.001)
    e_onset, is_onset = onset_energy_level(x, alpha=5e-5)
    e = np.append(e, e_onset)

    # collection update
    collector.update_collect(x.copy())
    
  # no data
  except queue.Empty:
    return x, e, is_onset

  return x, e, is_onset


def clear_mic_queue():
  """
  clear the queue after classification
  """

  # empty queue
  while not q.empty():
    dummy = q.get_nowait()


def stop_mic_condition(x, fs, time_duration):
  """
  stop the while loop
  """

  #return np.abs(start_time - stream.time) < time_duration
  return x.shape[0] < (time_duration * fs)


def sd_setup(fs, hop, fs_device=48000, channels=1):
  """
  init sound device
  """

  # determine downsample
  downsample = fs_device // fs

  # select microphone
  sd.default.device = 7

  # show devices
  print("\ndevice list: \n", sd.query_devices())

  # setup stream sounddevice
  stream = sd.InputStream(samplerate=fs*downsample, blocksize=hop*downsample, channels=channels, callback=callback_mic)

  return stream, downsample


if __name__ == '__main__':
  """
  ML - Machine Learning file
  """

  # path
  plot_path = './ignore/plots/mic/'

  # create folder
  create_folder([plot_path])

  # classifier
  classifier = Classifier(model_path='./models/conv-fstride/v3_c-5_n-2000/bs-32_it-1000_lr-1e-05/', model_file_name='model.pth', params_file_name='params.npz', verbose=True)


  # --
  # read mic input

  # params
  fs, time_duration = 16000, 10

  # window and hop size
  N, hop = int(0.025 * fs), int(0.010 * fs)

  # frame size of features
  frame_size = 32

  # feature extractor
  feature_extractor = FeatureExtractor(fs, N=N, hop=hop, n_filter_bands=32, n_ceps_coeff=12, frame_size=frame_size)


  global q, downsample

  # queue for audio samples
  q = queue.Queue()

  # input collector class
  collector = Collector(N=N, hop=hop, frame_size=frame_size, update_size=32, frames_post=32)

  # sound device
  stream, downsample = sd_setup(fs, hop)
  
  # data collection
  x = np.empty(shape=(0), dtype=np.float32)
  energy_list = np.empty(shape=(0), dtype=np.float32)
  onset_frames = np.empty(shape=(0), dtype=np.float32)

  # list
  y_hat_list = []

  time_feature_list = []
  time_class_list = []

  # stream and update
  with stream:

    # sleep caller
    sd.sleep(int(100))
    
    print("--recording for {}s ...".format(time_duration))

    # loop
    while stop_mic_condition(x, fs, time_duration):

      # read chunk
      xi, e, is_onset = read_mic_data(collector)

      # concatenate samples
      x = np.concatenate((x, xi))
      energy_list = np.concatenate((energy_list, e))

      # onset was detected
      if is_onset:

        # determine onset in frames
        onset_frame = len(x) / hop

        # save into onset frame collector
        onset_frames = np.append(onset_frames, onset_frame)

        # start collection of items
        collector.start_collecting()


      # collection is full
      if collector.is_full():

        print("que size: ", q.qsize())

        # read out collection
        x_onset = collector.read_collection()

        # start time
        start_time = time.time()

        # extract features
        mfcc_bon, bon_pos = feature_extractor.extract_mfcc39(x_onset)

        # times
        time_feature_list.append(time.time() - start_time)
        start_time = time.time()

        # classify collection
        y_hat = classifier.classify_sample(mfcc_bon)
        y_hat_list.append(y_hat)

        # times
        time_class_list.append(time.time() - start_time)
        start_time = time.time()

        
        # clear queue
        clear_mic_queue()
        print("que size: ", q.qsize())

        # plot profile
        plot_mfcc_profile(x_onset[bon_pos*hop:(bon_pos+frame_size)*hop], fs, N, hop, mfcc_bon, frame_size=frame_size, plot_path=plot_path, name='collect-{}'.format(collector.collection_counter))



  # some more prints
  print("\n--end of recording\nx: {}, x_energy: {}\ncollections: {}\nmean energy: {}\ny_hats: {}".format(x.shape, energy_list.shape, collector.collection_counter, np.mean(energy_list), y_hat_list))
  print("\ntimes feat: {}\ntimes class: {}".format(time_feature_list, time_class_list))
  print("\ntimes feat: {:.4f}\ntimes class: {:.4f}".format(np.mean(time_feature_list), np.mean(time_class_list)))
  plot_waveform(x, fs, energy_list * 10, hop, onset_frames, title='input stream', ylim=(-1, 1), plot_path=None, name='None')
  plt.show()



