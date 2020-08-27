"""
Online Microphone input classification
"""

import numpy as np
import matplotlib.pyplot as plt

import os
import torch
import queue

# sound stuff
import sounddevice as sd

# my stuff
from common import *
from plots import *
from conv_nets import *
from ml import get_nn_model
from feature_extraction import calc_mfcc39, calc_onsets, onset_energy_level
from collector import Collector
from mic import Mic


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
    e_onset, is_onset = onset_energy_level(x, alpha=0.0001)
    e = np.append(e, e_onset)

    # collection update
    collector.update_collect(x.copy())
    
  # no data
  except queue.Empty:
    return x, e, is_onset

  return x, e, is_onset


def stop_mic_condition(x, fs, time_duration):
  """
  stop the while loop
  """

  #return np.abs(start_time - stream.time) < time_duration
  return x.shape[0] < (time_duration * fs)


def classify_sample(model, x, class_dict, verbose=True):
  """
  classification by neural network
  """

  # input to tensor
  x = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(x.astype(np.float32)), 0), 0)

  # no gradients for eval
  with torch.no_grad():

    # classify
    o = model(x)

    # prediction
    _, y_hat = torch.max(o.data, 1)

    if verbose:
      print("\nnew sample:\nprediction: {} - {}\noutput: {}".format(y_hat, list(class_dict.keys())[list(class_dict.values()).index(int(y_hat))], o.data))


def extract_model(file):
  """
  extract model from info file
  """

  # data loading
  data = np.load(file, allow_pickle=True)

  # print info
  print("\nextract model with params: {}\nand class dict: {}".format(data['param_str'], data['class_dict']))
  
  # extract data
  nn_arch, class_dict, path_to_file = data['params'][()]['nn_arch'], data['class_dict'][()], str(data['model_file_path'])

  # init model
  model = get_nn_model(nn_arch, n_classes=len(class_dict))

  # load model
  model.load_state_dict(torch.load(path_to_file))

  # activate eval mode (no dropout layers)
  model.eval()

  return model, class_dict


def sd_setup(fs, hop, fs_device=48000, channels=1):
  """
  init sound device
  """

  # determine downsample
  downsample = fs_device // fs

  # select microphone
  sd.default.device = 7

  # show devices
  print("device list: \n", sd.query_devices())

  # setup stream sounddevice
  stream = sd.InputStream(samplerate=fs*downsample, blocksize=hop*downsample, channels=channels, callback=callback_mic)

  return stream, downsample


if __name__ == '__main__':
  """
  ML - Machine Learning file
  """

  # path
  plot_path,  model_path = './ignore/plots/mic/', './ignore/models/best_models/'

  # create folder
  create_folder([plot_path])

  # model name
  model_name = 'best_model_c-5.npz'

  # extract model from file data
  model, class_dict = extract_model(model_path + model_name)


  # --
  # read mic input

  # params
  fs, time_duration = 16000, 10

  # window and hop size
  N, hop = int(0.025 * fs), int(0.010 * fs)

  global q, downsample

  # queue for audio samples
  q = queue.Queue()

  # input collector class
  collector = Collector()

  # sound device
  stream, downsample = sd_setup(fs, hop)
  

  # data collection
  x = np.empty(shape=(0), dtype=np.float32)
  energy_list = np.empty(shape=(0), dtype=np.float32)
  onset_frames = np.empty(shape=(0), dtype=np.float32)

  # stream and update
  with stream:

    # sleep caller
    #sd.sleep(int(1 * 1000))
    
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
        #print("onset_frame: ", onset_frame)

        # save into onset frame collector
        onset_frames = np.append(onset_frames, onset_frame)

        # start collection of items
        collector.start_collecting()


      # collection is full
      if collector.is_full():

        # read out collection
        x_onset = collector.read_collection()

        # mfcc
        mfcc = calc_mfcc39(x_onset, fs, N=N, hop=hop, n_filter_bands=32, n_ceps_coeff=12)

        # plot profile
        plot_mfcc_profile(x_onset, fs, N, hop, mfcc, frame_size=32, plot_path=plot_path, name='collect-{}'.format(collector.collection_counter))

        # classify collection
        classify_sample(model, mfcc, class_dict)



  # some more prints
  print("\n--end of recording\nx: {}, x_energy: {}\ncollections: {}\nmean energy: {}".format(x.shape, energy_list.shape, collector.collection_counter, np.mean(energy_list)))

  plot_waveform(x, fs, energy_list * 10, hop, onset_frames, title='input stream', ylim=(-1, 1), plot_path=None, name='None')
  plt.show()



