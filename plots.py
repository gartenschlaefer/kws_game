"""
plot some figures
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_mfcc_profile(x, fs, mfcc, plot_path, name='None'):
  """
  plot mfcc extracted features from audio file
  mfcc: [m x l]
  """

  # time vector
  t = np.arange(0, len(x)/fs, 1/fs)

  # setup figure
  fig = plt.figure(figsize=(8, 8))

  # make a grid
  n_rows, n_cols, n_im_rows = 25, 20, 5
  gs = plt.GridSpec(n_rows, n_cols, wspace=0.4, hspace=0.3)

  # time series plot
  ax = fig.add_subplot(gs[0:n_im_rows-1, :n_cols-2])
  ax.plot(t, x)
  ax.grid()
  ax.set_title('time signal of ' + '"' + name + '"')
  ax.set_ylabel("magnitude")
  ax.set_xlim([0, t[-1]])

  # select mfcc coeffs in arrays
  sel_coefs = [np.arange(0, 12), np.arange(12, 24), np.arange(24, 36), np.arange(36, 39)]
  titles = ['12 MFCCs', 'deltas', 'double deltas', 'energies']

  # mfcc plots
  for i, c in enumerate(sel_coefs):

    # row start and stop
    rs = (i+1) * n_im_rows + 2
    re = (i+2) * n_im_rows

    # specify grid pos
    ax = fig.add_subplot(gs[rs:re, :n_cols-2])

    #im = ax.imshow(mfcc[c], aspect='auto', extent = [0, mfcc[c].shape[1], c[-1], c[0]])
    im = ax.imshow(mfcc[c], aspect='auto', extent = [0, t[-1], c[-1], c[0]])

    # color limited
    # if titles[i] != 'energies':
    #   im = ax.imshow(mfcc[c], aspect='auto', extent = [0, t[-1], c[-1], c[0]], vmin=-100, vmax=np.max(mfcc[c]))
    #
    # else:
    #   im = ax.imshow(mfcc[c], aspect='auto', extent = [0, t[-1], c[-1], c[0]])

    # some labels
    ax.set_title(titles[i])
    ax.set_ylabel("cepstrum coeff")
    if i == len(sel_coefs) - 1:
      ax.set_xlabel("time [s]")
    ax.set_xlim(left=0)

    # add colorbar
    ax = fig.add_subplot(gs[rs:re, n_cols-1])
    fig.colorbar(im, cax=ax)

  plt.savefig(plot_path + 'mfcc-' + name + '.png', dpi=150)
  plt.close()


def plot_mfcc_only(mfcc, fs=16000, hop=160, plot_path=None, name='None'):
  """
  plot mfcc extracted features only (no time series)
  mfcc: [m x l]
  """

  # get shape
  m, l = mfcc.shape

  # time vector
  t = np.arange(0, l * hop / fs, hop/fs)

  # setup figure
  fig = plt.figure(figsize=(8, 8))

  # make a grid
  n_rows, n_cols, n_im_rows = 20, 20, 5
  gs = plt.GridSpec(n_rows, n_cols, wspace=0.4, hspace=0.3)

  # select mfcc coeffs in arrays
  sel_coefs = [np.arange(0, 12), np.arange(12, 24), np.arange(24, 36), np.arange(36, 39)]
  titles = ['12 MFCCs' + ' of "' + name + '"', 'deltas', 'double deltas', 'energies']

  # mfcc plots
  for i, c in enumerate(sel_coefs):

    # row start and stop
    rs = (i) * n_im_rows + 2
    re = (i+1) * n_im_rows

    # specify grid pos
    ax = fig.add_subplot(gs[rs:re, :n_cols-2])

    # plot selected mfcc
    im = ax.imshow(mfcc[c], aspect='auto', extent = [0, t[-1], c[-1], c[0]])

    # some labels
    ax.set_title(titles[i])
    ax.set_ylabel("cepstrum coeff")
    if i == len(sel_coefs) - 1:
      ax.set_xlabel("time [s]")
    ax.set_xlim(left=0)

    # add colorbar
    ax = fig.add_subplot(gs[rs:re, n_cols-1])
    fig.colorbar(im, cax=ax)

  # plot the fig
  if plot_path is not None:
    plt.savefig(plot_path + 'mfcc-only-' + name + '.png', dpi=150)
    plt.close()

  # just show it
  else:
    plt.show()