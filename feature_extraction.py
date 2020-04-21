"""
feature extraction tools
"""
import numpy as np

def calc_mfcc(x, fs, N=1024, hop=512, n_filter_bands=8):
  """
  mel-frequency cepstral coefficient
  """

  # stft
  X = custom_stft(x, 2*N, hop)

  # weights
  w_f, w_mel, n_bands = mel_band_weights(n_filter_bands, fs, N)

  # energy of fft
  E = np.power(np.abs(X[:, 0:N]), 2)

  # sum the weighted energies
  u = np.inner(E, w_f)

  # discrete cosine transform of log
  return dct(np.log(u), n_bands).T


def dct(X, N):
  """
  discrete cosine transform
  """
  
  # transformation matrix
  H = np.cos(np.pi / N * np.outer((np.arange(N) + 0.5), np.arange(N)))

  # transformed signal
  return np.dot(X, H)


def triangle(M, N):
  """
  create a triangle
  """
  return np.concatenate((np.linspace(0, 1, M), np.linspace(1 - 1 / N, 0, N - 1)))


def mel_band_weights(n_bands, fs, N=1024, overlap=0.5):
  """
  mel_band_weights create a weight matrix of triangluar mel band weights for a filter bank.
  This is used to compute MFCC.
  """

  # hop of samples
  hop = N / (n_bands + 1)

  # calculating middle point of triangle
  mel_samples = np.arange(hop, N, hop)
  f_samples = np.round(mel_to_f(mel_samples / N * f_to_mel(fs / 2)) * N / (fs / 2))

  # round mel samples too
  mel_samples = np.round(mel_samples)

  # complicated hop sizes for frequency scale
  hop_f = (f_samples - np.roll(f_samples, +1))
  hop_f[0] = f_samples[0]

  # triangle shape
  tri = triangle(hop, hop+1)

  # weight init
  w_mel = np.zeros((n_bands, N))
  w_f = np.zeros((n_bands, N))

  for mi in range(n_bands):

    # for equidistant mel scale
    w_mel[mi][int(mel_samples[mi])] = 1
    w_mel[mi] = np.convolve(w_mel[mi, :], tri, mode='same')

    # for frequency scale
    w_f[mi, int(f_samples[mi])] = 1
    w_f[mi] = np.convolve(w_f[mi], triangle(hop_f[mi]+1, hop_f[mi]+1), mode='same')

  # print("w_f: ", f_samples.shape)
  # print("w_f: ", w_f.shape)

  # plt.figure(1)
  # plt.plot(w_f.T)
  # plt.show()

  return (w_f, w_mel, n_bands)


def custom_stft(x, N=1024, hop=512, norm=True):
  """
  short time fourier transform
  """
  # windowing
  w = np.hanning(N)

  # apply windows
  x_buff = np.multiply(w, create_frames(x, N, hop))

  # transformation matrix
  H = np.exp(1j * 2 * np.pi / N * np.outer(np.arange(N), np.arange(N)))

  # normalize if asked
  if norm:
    return 2 / N * np.dot(x_buff, H)

  # transformed signal
  return np.dot(x_buff, H)


def create_frames(x, N, hop):
  """
  create_frames from input 
  """

  # number of samples in window
  N = int(N)

  # number of windows
  win_num = (len(x) - N) // hop + 1 

  # remaining samples
  r = int(np.remainder(len(x), hop))
  if r:
    win_num += 1;

  # segments
  windows = np.zeros((win_num, N))

  # segmentation
  for wi in range(0, win_num):

    # remainder
    if wi == win_num - 1 and r:
      windows[wi] = np.concatenate((x[wi * hop :], np.zeros(N - len(x[wi * hop :]))))

    # no remainder
    else:
      windows[wi] = x[wi * hop : (wi * hop) + N]

  return windows