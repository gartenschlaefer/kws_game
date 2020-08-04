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


def callback_mic(indata, frames, time, status):
  """
  Input Stream Callback
  """

  if status:
    print(status)

  # put into queue
  q.put(indata[:, 0])


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

    #print("data: ", x.shape)

    # mean energy: 0.0002405075

    # detect onset
    e_onset, is_onset = onset_energy_level(x, alpha=0.001)
    e = np.append(e, e_onset)
    #print("energy: ", e)

    # collection update
    collector.update_collect(x)
    

  # no data
  except queue.Empty:
    return x, e, False

  return x, e, is_onset


def stop_mic_condition(x, fs, time_duration):
  """
  stop the while loop
  """

  #return np.abs(start_time - stream.time) < time_duration
  return x.shape[0] < (time_duration * fs)


def classify_sample(model, x):
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

    print("prediction: ", y_hat)


if __name__ == '__main__':
  """
  ML - Machine Learning file
  """

  # path
  plot_path,  model_path = './ignore/plots/mic/', './ignore/models/'
  
  # create folder
  create_folder([plot_path])

  # var selection for model
  nn_arch, version_id, n_examples_class, batch_size, num_epochs, lr = 'conv-trad', 2, 2000, 32, 200, 1e-4

  # init model
  model = get_nn_model(nn_arch)

  # param string
  param_str = '{}_v{}_n-{}_bs-{}_it-{}_lr-{}'.format(nn_arch, version_id, n_examples_class, batch_size, num_epochs, str(lr).replace('.', 'p'))

  # load model
  model.load_state_dict(torch.load(model_path + param_str + '.pth'))


  # --
  # read mic input

  # params
  fs, time_duration = 16000, 1

  # window and hop size
  N, hop = int(0.025 * fs), int(0.010 * fs)

  global q

  # queue for audio samples
  q = queue.Queue()

  # input collector class
  collector = Collector()
  

  # setup stream sounddevice
  stream = sd.InputStream(samplerate=fs, blocksize=hop, channels=1, callback=callback_mic)

  # data collection
  x = np.empty(shape=(0), dtype=np.float32)
  energy_list = np.empty(shape=(0), dtype=np.float32)
  onset_frames = np.empty(shape=(0), dtype=np.float32)

  # stream and update
  with stream:

    # sleep caller
    #sd.sleep(int(1 * 1000))
    
    print("recording for {}s ...".format(time_duration))
    # run forever
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
        print("yey full collection")

        # read out collection
        x_onset = collector.read_collection()

        # mfcc
        mfcc = calc_mfcc39(x_onset, fs, N=N, hop=hop, n_filter_bands=32, n_ceps_coeff=12)
        print("mfcc: ", mfcc.shape)

        # classify collection
        classify_sample(model, mfcc)



  
  print("x: ", x.shape)
  print("x: ", energy_list.shape)

  print("mean energy: ", np.mean(energy_list))


  plot_waveform(x, fs, energy_list * 10, hop, onset_frames, title='input stream', ylim=(-1, 1), plot_path=None, name='None')
  plt.show()



