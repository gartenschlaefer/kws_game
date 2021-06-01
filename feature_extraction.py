"""
feature extraction tools
"""

import numpy as np
import librosa
import scipy
import sys

from legacy import legacy_adjustments_feature_params
from skimage.util.shape import view_as_windows


class FeatureExtractor():
  """
  feature extractor class with MFCC features
  """

  def __init__(self, feature_params):

    # arguments
    self.feature_params = legacy_adjustments_feature_params(feature_params)

    # windowing params
    self.N, self.hop = int(self.feature_params['N_s'] * self.feature_params['fs']), int(self.feature_params['hop_s'] * self.feature_params['fs'])

    # energy calculation
    self.use_e_norm, self.use_e_sqrt = (False, True) if 'use_energy_features' in self.feature_params.keys() else (True, False)

    # position of energy vector (for energy region)
    self.energy_feature_pos = 0

    # channel size
    self.channel_size = 1 if not self.feature_params['use_channels'] else int(self.feature_params['use_cepstral_features']) + int(self.feature_params['use_delta_features']) +  int(self.feature_params['use_double_delta_features'])

    # feature size
    self.feature_size = (self.feature_params['n_ceps_coeff'] + int(self.feature_params['use_energy_features'])) * int(self.feature_params['use_cepstral_features']) + (self.feature_params['n_ceps_coeff'] + int(self.feature_params['use_energy_features'])) * int(self.feature_params['use_delta_features']) + (self.feature_params['n_ceps_coeff'] + int(self.feature_params['use_energy_features'])) * int(self.feature_params['use_double_delta_features']) if not self.feature_params['use_channels'] else (self.feature_params['n_ceps_coeff'] + int(self.feature_params['use_energy_features']))

    # frame size
    self.frame_size = self.feature_params['frame_size']

    # raw frame size for raw inputs in samples
    self.raw_frame_size = int(self.feature_params['frame_size_s'] * self.feature_params['fs'])

    # zero does not work
    if self.feature_size == 0 or self.channel_size == 0: print("feature size is zero -> select features in config"), sys.exit()

    # calculate weights
    self.w_f, self.w_mel, self.f, self.m = mel_band_weights(self.feature_params['n_filter_bands'], self.feature_params['fs'], self.N//2+1)


  def extract_mfcc(self, x, reduce_to_best_onset=True):
    """
    extract mfcc features fast return [c, m, n], best onset pos
    """

    # calculate mfcc
    mfcc = self.calc_mfcc(x)

    # mfcc collected
    mfcc_all = np.empty(shape=(1, 0, mfcc.shape[1]), dtype=np.float32) if self.channel_size == 1 else np.empty(shape=(0, self.feature_size, mfcc.shape[1]), dtype=np.float32)

    # compute deltas [feature x frames]
    deltas = self.compute_deltas(mfcc)

    # compute double deltas [feature x frames]
    double_deltas = self.compute_deltas(deltas)

    # compute energy of mfcc
    e_mfcc, e_deltas, e_double_deltas = self.calc_energy(mfcc), self.calc_energy(deltas), self.calc_energy(double_deltas)

    # stack as features
    if self.channel_size == 1:
      if self.feature_params['use_cepstral_features']: mfcc_all = np.concatenate((mfcc_all, mfcc[np.newaxis, :]), axis=1) if not self.feature_params['use_energy_features'] else np.concatenate((mfcc_all, mfcc[np.newaxis, :], e_mfcc[np.newaxis, :]), axis=1)
      if self.feature_params['use_delta_features']: mfcc_all = np.concatenate((mfcc_all, deltas[np.newaxis, :]), axis=1) if not self.feature_params['use_energy_features'] else np.concatenate((mfcc_all, deltas[np.newaxis, :], e_deltas[np.newaxis, :]), axis=1)
      if self.feature_params['use_double_delta_features']: mfcc_all = np.concatenate((mfcc_all, double_deltas[np.newaxis, :]), axis=1) if not self.feature_params['use_energy_features'] else np.concatenate((mfcc_all, double_deltas[np.newaxis, :], e_double_deltas[np.newaxis, :]), axis=1)

    # stack as channels
    else:
      if self.feature_params['use_cepstral_features']: mfcc_all = np.concatenate((mfcc_all, mfcc[np.newaxis, :]), axis=0) if not self.feature_params['use_energy_features'] else np.concatenate((mfcc_all, np.vstack((mfcc, e_mfcc))[np.newaxis, :]), axis=0)
      if self.feature_params['use_delta_features']: mfcc_all = np.concatenate((mfcc_all, deltas[np.newaxis, :]), axis=0) if not self.feature_params['use_energy_features'] else np.concatenate((mfcc_all, np.vstack((deltas, e_deltas))[np.newaxis, :]), axis=0)
      if self.feature_params['use_double_delta_features']: mfcc_all = np.concatenate((mfcc_all, double_deltas[np.newaxis, :]), axis=0) if not self.feature_params['use_energy_features'] else np.concatenate((mfcc_all, np.vstack((double_deltas, e_double_deltas))[np.newaxis, :]), axis=0)

    # norm -> [0, 1]
    if self.feature_params['norm_features']: 
      for i, m in enumerate(mfcc_all[0, :]): mfcc_all[0, i] = (m + np.abs(np.min(m))) / np.linalg.norm(m + np.abs(np.min(m)), ord=np.infty)

    # find best onset
    _, bon_pos = self.find_max_energy_region(mfcc[self.energy_feature_pos, :])

    # return best onset
    if reduce_to_best_onset: return mfcc_all[:, :, bon_pos:bon_pos+self.frame_size], bon_pos

    return mfcc_all, bon_pos


  def calc_energy(self, x):
    """
    energy calculation
    """
    e = np.einsum('ij,ji->j', x, x.T)
    if self.use_e_sqrt: e = np.sqrt(e)
    if self.use_e_norm: e = e / np.max(e)
    return e[np.newaxis, :]


  def calc_mfcc(self, x):
    """
    calculate mfcc
    """

    # pre processing
    x_pre = self.pre_processing(x)

    # stft
    X = 2 / self.N * librosa.stft(x_pre, n_fft=self.N, hop_length=self.hop, win_length=self.N, window='hann', center=False).T

    # energy of fft (one-sided)
    E = np.power(np.abs(X), 2)

    # sum the weighted energies
    u = np.inner(E, self.w_f)

    # mfcc
    mfcc = scipy.fftpack.dct(np.log(u), type=2, n=self.feature_params['n_filter_bands'], axis=1, norm=None, overwrite_x=False).T[:self.feature_params['n_ceps_coeff']]

    return mfcc


  def calc_spectogram(self, x):
    """
    spectogram (power spectrum)
    """

    # pre processing
    x_pre = self.pre_processing(x)

    # stft
    x_stft = 2 / self.N * librosa.stft(x_pre, n_fft=self.N, hop_length=self.hop, win_length=self.N, window='hann', center=False).T

    # power spectrum
    return np.abs(x_stft * np.conj(x_stft))


  def pre_processing(self, x):
    """
    actual preprocessing with dithering and normalization
    """
    
    # make a copy
    x = x.copy()

    # dither
    x = self.add_dither(x)

    # normalize input signal with infinity norm
    x = librosa.util.normalize(x)

    return x


  def add_dither(self, x):
    """
    add a dither signal
    """

    # determine abs min value except from zero, for dithering
    try:
      min_val = np.min(np.abs(x[np.abs(x)>0]))
    except:
      print("only zeros in this signal")
      min_val = 1e-4

    # add some dither
    x += np.random.normal(0, 0.5, len(x)) * min_val

    return x


  def compute_deltas(self, x):
    """
    compute deltas for mfcc [feature x frames]
    """

    # init
    d = np.zeros(x.shape)
    
    # zero-padding
    x_pad = np.pad(x, ((0, 0), (1, 1)))

    # for all time frames
    for t in range(x.shape[1]):
      
      # calculate diff
      d[:, t] = (x_pad[:, t+2] - x_pad[:, t]) / 2

    # clean first and last entry
    d[:, -1] = d[:, -2]
    d[:, 0] = d[:, 1]

    return d


  def find_max_energy_region(self, x, randomize=False, rand_frame=5):
    """
    find frame with least amount of energy
    """

    # windowed [r x m x f]
    x_win = np.squeeze(view_as_windows(x, self.feature_params['frame_size'], step=1))

    # energy region
    if self.energy_feature_pos == 0: bon_pos = np.argmax(np.sum(x_win, axis=1))
    else: bon_pos = np.argmin(np.sum(x_win, axis=1))

    # randomize a bit
    if randomize:
      bon_pos += np.random.randint(-rand_frame, rand_frame)
      if bon_pos >= x_win.shape[0]: bon_pos = x_win.shape[0]# - 1
      #if bon_pos > x_win.shape[0]-1: bon_pos = x_win.shape[0]-1
      elif bon_pos < 0: bon_pos = 0

    return frames_to_time(bon_pos, self.feature_params['fs'], self.hop), bon_pos


  def extract_mfcc39_slow(self, x):
    """
    extract mfcc features, slow implementation of my own - not used anymore
    """

    # pre processing
    x_pre = self.pre_processing(x)

    # stft
    X = custom_stft(x_pre, self.N, self.hop)

    # energy of fft (one-sided)
    E = np.power(np.abs(X[:, :self.N//2+1]), 2)

    # sum the weighted energies
    u = np.inner(E, self.w_f)

    # mfcc
    mfcc = (custom_dct(np.log(u), self.n_filter_bands).T)[:self.n_ceps_coeff]

    # compute deltas [feature x frames]
    deltas = self.compute_deltas(mfcc)

    # compute double deltas [feature x frames]
    double_deltas = self.compute_deltas(deltas)

    # compute energies [1 x frames]
    e_mfcc = np.vstack((
      np.sum(mfcc**2, axis=0) / np.max(np.sum(mfcc**2, axis=0)), 
      np.sum(deltas**2, axis=0) / np.max(np.sum(deltas**2, axis=0)), 
      np.sum(double_deltas**2, axis=0) / np.max(np.sum(double_deltas**2, axis=0))
      ))

    # stack and get best onset
    mfcc = np.vstack((mfcc, deltas, double_deltas, e_mfcc))

    # find best onset
    _, bon_pos = self.find_max_energy_region(mfcc[self.energy_feature_pos, :], self.fs, self.hop)

    # return best onset
    return mfcc[:, bon_pos:bon_pos+self.frame_size], bon_pos


  def get_best_raw_samples(self, x, randomize=False, rand_samples=10, add_channel_dim=False):
    """
    best raw samples computed upon energy
    """

    # search for max energy windowed [r x m]
    e_win = np.squeeze(view_as_windows(np.abs(x)**2, self.raw_frame_size, step=1))

    # energy region
    bon_pos = np.argmax(np.sum(e_win, axis=1))

    # randomize a bit
    if randomize:
      bon_pos += np.random.randint(-rand_frame, rand_frame)
      if bon_pos >= x_win.shape[0]: bon_pos = x_win.shape[0]
      elif bon_pos < 0: bon_pos = 0

    # best onset
    x = x[bon_pos:bon_pos+self.raw_frame_size]

    # channel dim
    if add_channel_dim: x = x[np.newaxis, :]

    return x, bon_pos


  def mu_softmax(self, x, mu=256):
    """
    mu softmax function
    """
    return np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + np.ones(x.shape) * mu)


  def quantize(self, x, quant_size=256):
    """
    quantize data
    """
    return np.digitize(self.mu_softmax(x, mu=quant_size), bins=np.linspace(-1, 1, quant_size)) - 1


  def invert_mfcc(self, mfcc):
    """
    invert mfcc
    """

    x = librosa.feature.inverse.mfcc_to_audio(mfcc, n_mels=32, dct_type=2, norm=None, ref=1.0, lifter=0, sr=self.feature_params['fs'], n_fft=self.N, hop_length=self.hop, window='hann')

    return x

# --
# other useful functions

def find_min_energy_time(mfcc, fs, hop):
  """
  find min  energy time position
  """
  return frames_to_time(np.argmin(mfcc[36, :]), fs, hop)


def find_best_onset(onsets, frame_size=32, pre_frames=1):
  """
  find the best onset with highest propability of spoken word
  """

  # init
  best_onset, bon_pos = np.zeros(onsets.shape), 0

  # determine onset positions
  onset_pos = np.squeeze(np.argwhere(onsets))

  # single onset handling
  if int(np.sum(onsets)) == 1:

    #return onsets, int(np.where(onsets == 1)[0][0])
    best_onset = onsets
    bon_pos = onset_pos

  # multiple onsets handling
  else: 

    # windowing
    o_win = view_as_windows(np.pad(onsets, (0, frame_size-1)), window_shape=(frame_size), step=1)[onset_pos, :]

    # get index of best onset
    x_max = np.argmax(np.sum(o_win, axis=1))

    # set single best onset
    bon_pos = onset_pos[x_max]
    best_onset[bon_pos] = 1

  # pre frames before real onset
  if bon_pos - pre_frames > 0:
    best_onset = np.roll(best_onset, -pre_frames)

  # best onset on right egde, do roll
  if bon_pos - pre_frames >= (onsets.shape[0] - frame_size):
    r = frame_size - (onsets.shape[0] - (bon_pos - pre_frames)) + 1
    best_onset = np.roll(best_onset, -r)

  #print("best_onset: ", best_onset)
  #print("pos: ", int(np.where(best_onset == 1)[0][0]))

  return best_onset, int(np.where(best_onset == 1)[0][0])


def frames_to_time(x, fs, hop):
  """
  transfer from frame space into time space (choose beginning of frame)
  """
  return x * hop / fs


def frames_to_sample(x, fs, hop):
  """
  frame to sample space
  """
  return x * hop


def calc_mfcc39(x, fs, N=400, hop=160, n_filter_bands=32, n_ceps_coeff=12, use_librosa=False):
  """
  calculate mel-frequency 39 feature vector
  """

  # get mfcc coeffs [feature x frames]
  if use_librosa:
    import librosa
    mfcc = librosa.feature.mfcc(x, fs, S=None, n_mfcc=n_filter_bands, dct_type=2, norm='ortho', lifter=0, n_fft=N, hop_length=hop, center=False)[:n_ceps_coeff]

  else:
    mfcc = custom_mfcc(x, fs, N, hop, n_filter_bands)[:n_ceps_coeff]
  
  # compute deltas [feature x frames]
  deltas = compute_deltas(mfcc)

  # compute double deltas [feature x frames]
  double_deltas = compute_deltas(deltas)

  # compute energies [1 x frames]
  e_mfcc = np.vstack((
    np.sum(mfcc**2, axis=0) / np.max(np.sum(mfcc**2, axis=0)), 
    np.sum(deltas**2, axis=0) / np.max(np.sum(deltas**2, axis=0)), 
    np.sum(double_deltas**2, axis=0) / np.max(np.sum(double_deltas**2, axis=0))
    ))

  return np.vstack((mfcc, deltas, double_deltas, e_mfcc))


def custom_mfcc(x, fs, N=1024, hop=512, n_filter_bands=8):
  """
  mel-frequency cepstral coefficient
  """

  # stft
  X = custom_stft(x, N, hop)

  # weights
  w_f, w_mel, _, _ = mel_band_weights(n_filter_bands, fs, N//2)

  # energy of fft (one-sided)
  E = np.power(np.abs(X[:, :N//2]), 2)

  # sum the weighted energies
  u = np.inner(E, w_f)

  # discrete cosine transform of log
  return custom_dct(np.log(u), n_filter_bands).T


def custom_dct_matrix(N):
  """
  get custom dct matrix
  """
  return np.cos(np.pi / N * np.outer((np.arange(N) + 0.5), np.arange(N)))


def custom_dct(x, N):
  """
  discrete cosine transform of matrix [MxN]
  """
  return np.dot(x, custom_dct_matrix(N))


def mel_to_f(m):
  """
  mel to frequency
  """
  return 700 * (np.power(10, m / 2595) - 1)


def f_to_mel(f):
  """
  frequency to mel 
  """
  return 2595 * np.log10(1 + f / 700)


def triangle(M, N, same=True):
  """
  create a triangle
  """

  # ensure int
  M, N = int(M), int(N)

  # triangle
  tri = np.concatenate((np.linspace(0, 1, M), np.linspace(1 - 1 / N, 0, N - 1)))

  # same amount of samples in M and N space -> use zero padding
  if same:

    # zeros to append
    k = M - N

    # zeros at beginning
    if k < 0: return np.pad(tri, (int(np.abs(k)), 0))

    # zeros at end
    else: return np.pad(tri, (0, int(np.abs(k))))

  return tri


def mel_band_weights(n_bands, fs, N=1024):
  """
  mel_band_weights create a weight matrix of triangular Mel band weights for a filter bank.
  This is used to compute MFCC.
  """

  # hop of samples
  hop = (N - 1) / (n_bands + 1)

  # the scales
  mel_scale = np.linspace(0, f_to_mel(fs / 2), N)
  f_scale = mel_to_f(mel_scale)

  # calculating middle point of triangle
  mel_samples = np.arange(hop, N + n_bands, hop) - 1
  f_samples = np.round(mel_to_f(mel_samples / N * f_to_mel(fs / 2)) * N / (fs / 2))

  # round mel samples too
  mel_samples = np.round(mel_samples)

  # last entry, account for rounding errors
  mel_samples[-1] = N - 1
  f_samples[-1] = N - 1

  # diff
  hop_m = np.insert(np.diff(mel_samples), 0, mel_samples[0])
  hop_f = np.insert(np.diff(f_samples), 0, f_samples[0])

  # weight init
  w_mel = np.zeros((n_bands, N))
  w_f = np.zeros((n_bands, N))

  for mi in range(n_bands):

    # for equidistant mel scale
    w_mel[mi][int(mel_samples[mi])] = 1
    w_mel[mi] = np.convolve(w_mel[mi, :], triangle(hop_m[mi]+1, hop_m[mi+1]+1), mode='same')

    # for frequency scale
    w_f[mi, int(f_samples[mi])] = 1
    w_f[mi] = np.convolve(w_f[mi], triangle(hop_f[mi]+1, hop_f[mi+1]+1), mode='same')

  return (w_f, w_mel, f_scale, mel_scale)


def calc_onsets(x, fs, N=1024, hop=512, adapt_frames=5, adapt_alpha=0.1, adapt_beta=1):
  """
  calculate onsets with complex domain and adapt thresh
  """

  # stft
  X = custom_stft(x, N=N, hop=hop, norm=True)

  # complex domain
  c = complex_domain_onset(X, N)

  # adaptive threshold
  thresh = adaptive_threshold(c, H=adapt_frames, alpha=adapt_alpha, beta=adapt_beta)

  # get onsets from measure and threshold
  onsets = thresholding_onset(c, thresh)

  return onsets


def onsets_to_onset_times(onsets, fs, N, hop):
  """
  use onset vector [0, 0, 1, 0, 0, ...] and 
  create time vector [0.25, ...]
  """

  onset_times = (onsets * np.arange(0, len(onsets)) * hop + N / 2) / fs 
  return onset_times[onset_times > N / 2 / fs]


def thresholding_onset(x, thresh):
  """
  thresholding for onset events
  params: 
    x - input sequence
    thresh - threshold vector
  """

  # init
  onset = np.zeros(len(x))

  # set to one if over threshold
  onset[x > thresh] = 1

  # get only single onset -> attention edge problems
  onset = onset - np.logical_and(onset, np.roll(onset, 1))

  return onset


def adaptive_threshold(g, H=10, alpha=0.05, beta=1):
  """
  adaptive threshold with sliding window
  """

  # threshold
  thresh = np.zeros(len(g))

  # sliding window
  for i in np.arange(H//2, len(g) - H//2):

    # median thresh
    thresh[i] = np.median(g[i - H//2 : i + H//2])

  # linear mapping
  thresh = alpha * np.max(thresh) + beta * thresh

  return thresh


def complex_domain_onset(X, N):
  """
  complex domain approach for onset detection
  params:
    X - fft
    N - window size
  """

  # calculate phase deviation
  d = phase_deviation(X, N)

  # ampl target
  R = np.abs(X[:, 0:N//2])

  # ampl prediction
  R_h = np.roll(R, 1, axis=0)

  # complex measure
  gamma = np.sqrt(np.power(R_h, 2) + np.power(R, 2) - 2 * R_h * R * np.cos(d))

  # clean up first two indices
  gamma[0] = np.zeros(gamma.shape[1])

  # sum all frequency bins
  eta = np.sum(gamma, axis=1)

  return eta


def phase_deviation(X, N):
  """
  phase_deviation of STFT
  """

  # get unwrapped phase
  phi0 = np.unwrap(np.angle(X[:, 0:N//2]))
  phi1 = np.roll(phi0, 1, axis=0)
  phi2 = np.roll(phi0, 2, axis=0)

  # calculate phase derivation
  d = princarg(phi0 - 2 * phi1 + phi2)

  # clean up first two indices
  d[0:2] = np.zeros(d.shape[1])

  return d


def princarg(p):
  """
  principle argument
  """
  return np.mod(p + np.pi, -2 * np.pi) + np.pi


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



if __name__ == '__main__':
  """
  main file of feature extraction and how to use it
  """
  
  import yaml
  import time
  import matplotlib.pyplot as plt
  import librosa
  import librosa.display
  import soundfile

  from glob import glob
  from common import create_folder
  from plots import plot_mfcc_profile, plot_waveform

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # init feature extractor
  feature_extractor = FeatureExtractor(cfg['feature_params'])

  # wav dir
  wav_dir = './ignore/my_recordings/showcase_wavs/'
  #wav_dir = './' + cfg['datasets']['speech_commands']['plot_paths']['damaged_files']

  # annotation dir
  anno_dir = './ignore/my_recordings/showcase_wavs/annotation/'

  # analyze some wavs
  for wav, anno in zip(glob(wav_dir + '*.wav'), glob(anno_dir + '*.TextGrid')):

    # info
    print("\nwav: ", wav), print("anno: ", anno)

    # load audio
    x, _ = librosa.load(wav, sr=16000)

    # feature extraction
    mfcc, bon_pos = feature_extractor.extract_mfcc(x, reduce_to_best_onset=False)

    print("mfcc: ", mfcc.shape)
    
    # invert mfcc
    x_hat = feature_extractor.invert_mfcc(np.squeeze(mfcc))

    # save invert mfcc
    soundfile.write(wav.split('.wav')[0] + '_inv_mfcc.wav', x_hat, 16000, subtype=None, endian=None, format=None, closefd=True)

    print("x_hat: ", x_hat.shape)

    plot_mfcc_profile(x, 16000, feature_extractor.N, feature_extractor.hop, mfcc, anno_file=anno, sep_features=True, diff_plot=False, bon_pos=bon_pos, frame_size=cfg['feature_params']['frame_size'], plot_path=wav_dir, name=wav.split('/')[-1].split('.')[0], show_plot=False, close_plot=False)
    plot_waveform(x, 16000, anno_file=anno, hop=feature_extractor.hop, title=wav.split('/')[-1].split('.')[0]+'_my', plot_path=wav_dir, name=wav.split('/')[-1].split('.')[0], show_plot=False)
    
  # random
  #x = np.random.randn(16000)
  #mfcc, bon_pos = feature_extractor.extract_mfcc(x, reduce_to_best_onset=False)
  #plot_mfcc_profile(x, 16000, feature_extractor.N, feature_extractor.hop, mfcc, bon_pos=bon_pos, name='rand', show_plot=True)






