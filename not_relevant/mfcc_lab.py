"""
mfcc test functions
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import librosa.display
import soundfile


def some_test_signal(fs, t=1, f=500, sig_type='modulated', save_to_file=False):
  """
  test signal adapted from https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html
  """

  # samples
  samples = np.linspace(0, t, int(fs * t), endpoint=False)

  # modulated signal
  if sig_type == 'modulated':

    # amplitude and noise power
    amp = 2 * np.sqrt(2)
    noise_power = 0.01 * fs / 2

    # modulated signal
    mod = 500 * np.cos(2 * np.pi * 0.25 * samples)
    carrier = amp * np.sin(2 * np.pi * f * samples + mod)

    # noise
    noise = np.random.normal(scale=np.sqrt(noise_power), size=samples.shape)
    noise *= np.exp(-samples / 5)

    # synthesize signal
    x = carrier + noise

  # pure sine
  elif sig_type == 'sine':

    # create sine
    x = signal = np.sin(2 * np.pi * f * samples)

  # use random
  else:
    x = np.random.randn(int(fs * t))

  # save file
  if save_to_file:
    soundfile.write('./ignore/features/test.wav', x, fs, subtype=None, endian=None, format=None, closefd=True)

  return x


def test_some_stfts(x, fs, N, hop):
  """
  test some stft functions, plot from librosa website
  """

  stft_cus = custom_stft(x, N=N, hop=hop, norm=True)[:, :N//2]
  stft_lib = 2 / N * librosa.stft(x, n_fft=N, hop_length=hop, win_length=N, window='hann', center=False, dtype=None, pad_mode='reflect')[:N//2]
  f, t, stft_sci = scipy.signal.stft(x, fs=1.0, window='hann', nperseg=N, noverlap=N-hop, nfft=N, detrend=False, return_onesided=True, boundary='zeros', padded=False, axis=- 1)

  print("cus_stft: ", stft_cus.shape)
  print("lib_stft: ", stft_lib.shape)
  print("sci_stft: ", stft_sci.shape)

  # plot
  fig, ax = plt.subplots()
  img = librosa.display.specshow(librosa.amplitude_to_db(stft_cus.T, ref=np.max), sr=fs, hop_length=hop, y_axis='log', x_axis='time', ax=ax)
  ax.set_title('Power spectrogram cus')
  fig.colorbar(img, ax=ax, format="%+2.0f dB")

  fig, ax = plt.subplots()
  img = librosa.display.specshow(librosa.amplitude_to_db(stft_lib, ref=np.max), sr=fs, hop_length=hop, y_axis='log', x_axis='time', ax=ax)
  ax.set_title('Power spectrogram')
  fig.colorbar(img, ax=ax, format="%+2.0f dB")

  fig, ax = plt.subplots()
  img = librosa.display.specshow(librosa.amplitude_to_db(stft_sci, ref=np.max), sr=fs, hop_length=hop, y_axis='log', x_axis='time', ax=ax)
  ax.set_title('Power spectrogram')
  fig.colorbar(img, ax=ax, format="%+2.0f dB")

  plt.show()


def test_some_dcts(u, n_filter_bands, n_ceps_coeff):
  """
  test some dct functions, plot from librosa website
  """

  # test dct functions
  mfcc_custom = custom_dct(np.log(u), n_filter_bands).T[:n_ceps_coeff]
  mfcc_sci = scipy.fftpack.dct(np.log(u), type=2, n=n_filter_bands, axis=1, norm=None, overwrite_x=False).T[:n_ceps_coeff]

  print("mfcc_custom: ", mfcc_custom.shape)
  print("mfcc_sci: ", mfcc_sci.shape)

  plt.figure()
  librosa.display.specshow(mfcc_custom, x_axis='linear')
  plt.ylabel('DCT function')
  plt.title('DCT filter bank')
  plt.colorbar()
  plt.tight_layout()

  plt.figure()
  librosa.display.specshow(mfcc_sci, x_axis='linear')
  plt.ylabel('DCT function')
  plt.title('DCT filter bank')
  plt.colorbar()
  plt.tight_layout()
  plt.show()


def test_some_mfccs(x, fs, N, hop, n_filter_bands, n_ceps_coeff):
  """
  test some mfcc functions, plot from: https://librosa.org/doc/main/generated/librosa.feature.mfcc.html
  """

  mfcc_cus = calc_mfcc39(x, fs, N=N, hop=hop, n_filter_bands=n_filter_bands, n_ceps_coeff=n_ceps_coeff, use_librosa=False)
  mfcc_lib = calc_mfcc39(x, fs, N=N, hop=hop, n_filter_bands=n_filter_bands, n_ceps_coeff=n_ceps_coeff, use_librosa=True)

  print("mfcc_cus", mfcc_cus.shape)
  print("mfcc_lib", mfcc_lib.shape)

  fig, ax = plt.subplots()
  img = librosa.display.specshow(mfcc_cus, x_axis='time', ax=ax)
  fig.colorbar(img, ax=ax)
  ax.set(title='MFCC_cus')

  fig, ax = plt.subplots()
  img = librosa.display.specshow(mfcc_lib, x_axis='time', ax=ax)
  fig.colorbar(img, ax=ax)
  ax.set(title='MFCC_lib')

  plt.show()


def time_measurements(x, u, feature_params):
  """
  time measurements
  """

  # create feature extractor
  feature_extractor = FeatureExtractor(feature_params)

  # n measurements
  delta_time_list = []

  for i in range(100):

    # measure extraction time - start
    start_time = time.time()

    # time: 0.030081419944763182
    #y = calc_mfcc39(x, fs, N=400, hop=160, n_filter_bands=32, n_ceps_coeff=12, use_librosa=False)
  
    # time: 0.009309711456298829
    #y = calc_mfcc39(x, fs, N=400, hop=160, n_filter_bands=32, n_ceps_coeff=12, use_librosa=True)

    # time: 0.00014737367630004883
    #y = (custom_dct(np.log(u), n_filter_bands).T)

    # time: 6.929159164428711e-05
    #y = scipy.fftpack.dct(np.log(u), type=2, n=n_filter_bands, axis=1, norm=None, overwrite_x=False).T

    # time: 0.00418839693069458 *** winner
    y, _ = feature_extractor.extract_mfcc(x)
    
    # time: 0.015525884628295898
    #y, _ = feature_extractor.extract_mfcc39_slow(x)

    # time: 0.011266257762908936s
    #y = custom_stft(x, N=N, hop=hop, norm=True)

    # time: 0.0005800390243530274s
    #y = 2 / N * librosa.stft(x, n_fft=N, hop_length=hop, win_length=N, window='hann', center=True, dtype=None, pad_mode='reflect')

    # time: 0.00044193744659423826s
    #_, _, y = scipy.signal.stft(x, fs=1.0, window='hann', nperseg=N, noverlap=N-hop, nfft=N, detrend=False, return_onesided=True, boundary='zeros', padded=False, axis=- 1)

    # result of measured time diff
    delta_time_list.append(time.time() - start_time)

  # data shpae
  print("y: ", y.shape)

  # times
  print("delta_time: ", np.mean(delta_time_list))


def time_measure_callable(x, callback_f):
  """
  time measurement with callable
  """

  # n measurements
  delta_time_list = []

  for i in range(100):

    # measure extraction time - start
    start_time = time.time()

    # callable function
    callback_f(x)

    # result of measured time difference
    delta_time_list.append(time.time() - start_time)

  # times
  print("f: [{}] mean time: [{:.4e}]".format(callback_f.__name__, np.mean(delta_time_list)))


def energy_with_sum(x):
  """
  energy calculation with sum
  """
  #e = np.sum(x**2, axis=0)
  e = np.sqrt(np.sum(x**2, axis=0))
  return e / np.max(e)


def energy_with_matrix(x):
  """
  matrix calculation
  """
  #e = np.diag(x.T @ x)
  e = np.sqrt(np.diag(x.T @ x))
  return e / np.max(e)


def energy_einsum(x):
  """
  einsum energy
  """
  #e = np.einsum('ij,ji->j', x, x.T)
  e = np.sqrt(np.einsum('ij,ji->j', x, x.T))
  return e / np.max(e)


def power_spec_naive(x):
  """
  power spec calc
  """
  return np.power(np.abs(x), 2)


def power_spec_conj(x):
  """
  power spec calc
  """
  return np.abs(x * np.conj(x))



if __name__ == '__main__':
  """
  main file of feature extraction and how to use it
  """
  
  import yaml

  import sys
  sys.path.append("../")

  from common import create_folder
  from feature_extraction import FeatureExtractor

  # plot path
  #plot_path = './ignore/plots/fe/'

  # create folder
  #create_folder([plot_path])

  # yaml config file
  cfg = yaml.safe_load(open("../config.yaml"))


  # init feature extractor
  feature_extractor = FeatureExtractor(cfg['feature_params'])


  # --
  # params

  fs = 16000
  N = 400
  hop = 160
  n_filter_bands = 16
  n_ceps_coeff = 12


  # --
  # test signal

  # generate test signal
  x = some_test_signal(fs, t=1, save_to_file=False)

  # stft
  x_stft = 2 / N * librosa.stft(x, n_fft=N, hop_length=hop, win_length=N, window='hann', center=False).T

  # mfcc
  mfcc, _ = feature_extractor.extract_mfcc(x)
  if len(mfcc.shape) == 3: mfcc = np.squeeze(mfcc.reshape(1, -1, mfcc.shape[2]))

  print("mfcc: ", mfcc.shape)

  print("e_sum: ", energy_with_sum(mfcc)), print("e_m: ", energy_with_matrix(mfcc)), print("e_e: ", energy_einsum(mfcc))
  #print("pn: ", power_spec_naive(x_stft)), print("pc: ", power_spec_conj(x_stft))

  print("\ntime measures: ")
  time_measure_callable(mfcc, energy_einsum), time_measure_callable(mfcc, energy_with_sum), time_measure_callable(mfcc, energy_with_matrix)
  #time_measure_callable(x_stft, power_spec_naive), time_measure_callable(x_stft, power_spec_conj)


  # --
  # workflow

  # # create mel bands
  # w_f, w_mel, f, m = mel_band_weights(n_bands=n_filter_bands, fs=fs, N=N//2+1)

  # # stft
  # X = 2 / N * librosa.stft(x, n_fft=N, hop_length=hop, win_length=N, window='hann', center=False).T

  # # energy of fft (one-sided)
  # E = np.power(np.abs(X), 2)

  # # sum the weighted energies
  # u = np.inner(E, w_f)

  # # mfcc
  # mfcc = custom_dct(np.log(u), n_filter_bands).T

  # print("mfcc: ", mfcc.shape)


  # inverse mfcc
  # y = librosa.feature.inverse.mfcc_to_audio(mfcc, n_mels=32, dct_type=2, norm='ortho', ref=1.0)
  # print("x: ", x.shape)
  # print("y: ", y.shape)
  # soundfile.write('./ignore/features/inv_mfcc.wav', y, fs, subtype=None, endian=None, format=None, closefd=True)


  #--
  # test some functions

  #test_some_stfts(x, fs, N, hop)
  #test_some_dcts(u, n_filter_bands, n_ceps_coeff)
  #test_some_mfccs(x, fs, N, hop, n_filter_bands, n_ceps_coeff)


  # --
  # other stuff

  # time measurement
  #time_measurements(x, u, cfg['feature_params'])
