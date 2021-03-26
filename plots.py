"""
plot some figures
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from feature_extraction import onsets_to_onset_times, frames_to_time
from glob import glob
from praatio import tgio


def plot_other_grid(x, grid_size=(8, 8), title='grid1', plot_path=None, name='grid1', show_plot=False):
  """
  plot mfcc extracted features from audio file
  mfcc: [m x l]
  """

  # make a grid
  n_im_rows, n_im_cols = x.shape[2], x.shape[3]
  n_rows, n_cols = n_im_rows * grid_size[1] + grid_size[1]+1, n_im_cols * grid_size[0] + grid_size[0]+1

  fig = plt.figure(figsize=(8*n_cols/n_rows-0.5, 8))
  gs = plt.GridSpec(n_rows, n_cols, wspace=0.4, hspace=0.3)
  plt.title(title)
  plt.axis("off")

  i, j = 0, 0

  # mfcc plots
  for xi in x:

    # row start and stop
    rs = j * n_im_rows + 1
    re = (j + 1) * n_im_rows
    cs = i * n_im_cols + 1
    ce = (i + 1) * n_im_cols

    # exception for one row images
    if rs == re: re += 1

    # update indices
    if not i % (grid_size[0]-1) and i: i, j = 0, j + 1
    else: i += 1

    # specify grid pos
    ax = fig.add_subplot(gs[rs:re, cs:ce])

    # plot image coeffs
    im = ax.imshow(xi[0], aspect='equal')
    ax.axis("off")

  # TODO:
  # add colorbar
  #ax = fig.add_subplot(gs[rs:re, n_cols-1])
  #fig.colorbar(im, cax=ax)

  # plot save and show
  if plot_path is not None: 
    plt.savefig(plot_path + name + '.png', dpi=150)
    plt.close()

  if show_plot: plt.show()


def plot_grid_images(x, padding=1, num_cols=8, title='grid2', plot_path=None, name='grid2', show_plot=False):
  """
  plot grid images
  """

  # cast to numpy
  x = np.array(x)

  # get dimensions
  n_kernels, n_channels, n_features, n_frames = x.shape

  # determine minimum value
  value_min = np.min(x)

  # init images
  row_imgs = np.empty((n_channels, n_features, 0), dtype=x.dtype)
  grid_img = np.empty((n_channels, 0, n_frames * num_cols + num_cols + padding), dtype=x.dtype)
  all_grid_img = np.empty((0, n_frames * num_cols + num_cols + padding), dtype=x.dtype)

  # padding init
  v_padding = np.ones((n_channels, n_features, 1)) * value_min
  h_padding = np.ones((n_channels, 1, n_frames * num_cols + num_cols + padding)) * value_min

  # vars
  row, col = 0, 0

  # add each image
  for w in x:

    # horizontal filling
    row_imgs = np.concatenate((row_imgs, v_padding, w), axis=2)
    col += 1

    # max row imgs
    if col >= num_cols:

      row_imgs = np.concatenate((row_imgs, v_padding), axis=2)
      grid_img = np.concatenate((grid_img, h_padding, row_imgs), axis=1)
      row_imgs = np.empty((n_channels, n_features, 0), dtype=x.dtype)
      col = 0

  # rest of row
  if row_imgs.shape[2] != 0:
    row_imgs = np.concatenate((row_imgs, np.ones((n_channels, n_features, n_frames * num_cols + num_cols + padding - row_imgs.shape[2])) * value_min), axis=2)
    grid_img = np.concatenate((grid_img, h_padding, row_imgs), axis=1)
    row_imgs = np.empty((n_channels, n_features, 0), dtype=x.dtype)

  # last padding
  grid_img = np.concatenate((grid_img, h_padding), axis=1)

  # get all channels together
  for ch_grid_img in grid_img:
    all_grid_img = np.concatenate((all_grid_img, ch_grid_img), axis=0)

  # plot init
  fig = plt.figure(figsize=(np.clip(8 * n_frames//n_features, 4, 16), 8))

  # image
  ax = plt.axes()
  im = ax.imshow(all_grid_img, aspect='equal', interpolation='none')

  # design
  plt.axis("off")
  plt.title(title)

  # colorbar
  cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
  plt.colorbar(im, cax=cax)


  # plot save and show
  if plot_path is not None: 
    plt.savefig(plot_path + name + '.png', dpi=150)
    plt.close()
    
  if show_plot: plt.show()

  # torch grid (must be torch tensor)
  # import torchvision.utils as vutils
  # img = np.transpose(vutils.make_grid(weights['conv1'], padding=1, normalize=True), (1, 2, 0))
  # plt.figure(figsize=(8,8))
  # plt.axis("off")
  # plt.title("conv1")
  # plt.imshow(img)
  # plt.show()


def plot_damaged_file_score(z, plot_path=None, name='z_score', enable_plot=False):
  """
  damaged file score
  """

  # no plot generation
  if plot_path is None or enable_plot is False:
    return

  # setup figure
  fig = plt.figure(figsize=(9, 5))
  plt.scatter(x=z, y=np.zeros(len(z)))
  plt.ylabel('nothing')
  plt.xlabel('score')
  plt.grid()

  # plot the fig
  if plot_path is not None:
    plt.savefig(plot_path + name + '.png', dpi=150)
    plt.close()


def plot_waveform(x, fs, e=None, hop=None, onset_frames=None, title='none', xlim=None, ylim=None, plot_path=None, name='None'):
  """
  just a simple waveform
  """

  # time vector
  t = np.arange(0, len(x)/fs, 1/fs)

  # setup figure
  fig = plt.figure(figsize=(9, 5))
  plt.plot(t, x)

  # energy plot
  if e is not None:
    plt.plot(np.arange(0, len(x)/fs, 1/fs * hop), e)

  # draw onsets
  if onset_frames is not None:
    for onset in frames_to_time(onset_frames, fs, hop):
      plt.axvline(x=float(onset), dashes=(5, 1), color='k')

  plt.title(title)
  plt.ylabel('magnitude')
  plt.xlabel('time [s]')

  if xlim is not None:
    plt.xlim(xlim)

  if ylim is not None:
    plt.ylim(ylim)

  plt.grid()

  # plot the fig
  if plot_path is not None:
    plt.savefig(plot_path + name + '.png', dpi=150)
    plt.close()


def plot_onsets(x, fs, N, hop, onsets, title='none', plot_path=None, name='None'):
  """
  plot waveform with onsets
  """

  # plot the waveform
  plot_waveform(x, fs, title=title)
  
  # care for best onset
  onset_times = onsets_to_onset_times(onsets, fs, N, hop) 

  # draw onsets
  for onset in onset_times:
    plt.axvline(x=float(onset), dashes=(5, 1), color='k')

  # plot the fig
  if plot_path is not None:
    plt.savefig(plot_path + name + '.png', dpi=150)
    plt.close()


def plot_confusion_matrix(cm, classes, plot_path=None, name='None'):
  """
  plot confusion matrix
  """

  if cm is None:
    return

  # init plot
  fig, ax = plt.subplots( figsize=(8, 8) )
  im = ax.imshow(cm, cmap=plt.cm.Blues)

  # text handling
  for predict in range(len(classes)):
    for true in range(len(classes)):

      # color handling
      if predict != true:
        font_color = 'black'
      else:
        font_color = 'white'

      if len(classes) > 10:
        fontsize = 7
      else:
        fontsize = 10

      # write nums inside
      text = ax.text(true, predict, cm[predict, true], ha='center', va='center', color=font_color, fontsize=fontsize)

  # care about labels
  ax.set_xticks(np.arange(len(classes)))
  ax.set_yticks(np.arange(len(classes)))
  ax.set_xticklabels(classes)
  ax.set_yticklabels(classes)

  plt.xticks(fontsize=10, rotation=90)
  plt.yticks(fontsize=10, rotation=0)

  plt.xlabel('predicted labels')
  plt.ylabel('true labels')

  # plot the fig
  if plot_path is not None:
    plt.savefig(plot_path + name + '.png', dpi=150)
    plt.close()


def plot_train_score(train_score, plot_path, name_ext=''):
  """
  plot train scores
  """

  # usual loss in not adversarial nets
  if not train_score.is_adv:
    plot_train_loss(train_score.train_loss, train_score.val_loss, plot_path=plot_path, name='train_loss' + name_ext)
    plot_val_acc(train_score.val_acc, plot_path=plot_path, name='val_acc' + name_ext)

  # for adversarial nets
  else:
    plot_adv_train_loss(g_loss_fake=train_score.g_loss_fake, d_loss_fake=train_score.d_loss_fake, d_loss_real=train_score.d_loss_real, plot_path=plot_path, name='train_loss' + name_ext)


def plot_adv_train_loss(g_loss_fake, d_loss_fake, d_loss_real, plot_path=None, name='train_loss'):
  """
  train loss for adversarial networ
  """

  # setup figure
  fig = plt.figure(figsize=(8, 5))
  plt.plot(g_loss_fake, label='g_loss_fake')
  plt.plot(d_loss_fake, label='d_loss_fake')
  plt.plot(d_loss_real, label='d_loss_real')
  plt.ylabel("loss")
  plt.xlabel("iterations")
  plt.legend()
  plt.grid()

  # plot the fig
  if plot_path is not None:
    plt.savefig(plot_path + name + '.png', dpi=150)
    plt.close()


def plot_train_loss(train_loss, val_loss, plot_path=None, name='None'):
  """
  plot train vs. validation loss
  """

  # normalize
  if np.linalg.norm(train_loss, ord=np.infty):
    train_loss = train_loss / np.linalg.norm(train_loss, ord=np.infty)

  if np.linalg.norm(val_loss, ord=np.infty):
    val_loss = val_loss / np.linalg.norm(val_loss, ord=np.infty)

  # setup figure
  fig = plt.figure(figsize=(8, 5))
  plt.plot(train_loss, label='train loss')
  plt.plot(val_loss, label='val loss')
  plt.ylabel("normalized loss")
  plt.xlabel("iterations")
  #plt.ylim([0, 1])
  plt.legend()
  plt.grid()

  # plot the fig
  if plot_path is not None:
    plt.savefig(plot_path + name + '.png', dpi=150)
    plt.close()


def plot_val_acc(val_acc, plot_path=None, name='None'):
  """
  plot train vs. validation loss
  """

  # setup figure
  fig = plt.figure(figsize=(8, 5))
  plt.plot(val_acc, label='val acc')
  plt.ylabel("accuracy")
  plt.xlabel("iterations")
  plt.legend()
  plt.grid()

  # plot the fig
  if plot_path is not None:
    plt.savefig(plot_path + name + '.png', dpi=150)
    plt.close()


def plot_textGrid_annotation(anno_file, x=None, plot_text=False):
  """
  annotation
  """

  # annotation
  if anno_file is not None:

    # open annotation file
    tg = tgio.openTextgrid(anno_file)

    # get tier
    tier = tg.tierDict[tg.tierNameList[0]]

    # calculate height of text
    if x is not None and plot_text: h = np.min(x)
    else: h = 0

    # go through all entries
    for s, e, l in tier.entryList:
      plt.axvline(x=s, dashes=(3, 3), color='k', lw=1)
      if plot_text: plt.text(s + 0.01, h, l, color='k')


def plot_mfcc_profile(x, fs, N, hop, mfcc, anno_file=None, onsets=None, bon_pos=None, mient=None, minreg=None, frame_size=32, plot_path=None, name='mfcc_profile', enable_plot=False):
  """
  plot mfcc extracted features from audio file
  mfcc: [m x l]
  """

  # no plot generation
  if plot_path is None or enable_plot is False:
    return

  # time vector
  t = np.arange(0, len(x)/fs, 1/fs)

  # setup figure
  fig = plt.figure(figsize=(8, 8))

  # make a grid
  n_rows, n_cols, n_im_rows = 25, 20, 5
  gs = plt.GridSpec(n_rows, n_cols, wspace=0.4, hspace=0.3)

  # time series plot
  ax = fig.add_subplot(gs[0:n_im_rows-1, :n_cols-2])
  ax.plot(t[:len(x)], x)

  # annotation
  plot_textGrid_annotation(anno_file, x, plot_text=True)

  # min energy time and region
  if mient is not None:
    plt.axvline(x=mient, dashes=(5, 5), color='r', lw=2)
  if minreg is not None:
    plt.axvline(x=minreg, dashes=(5, 5), color='g', lw=2)

  # onset marks
  if onsets is not None:
    onset_times = onsets_to_onset_times(onsets, fs, N, hop)
    for onset in onset_times:
      plt.axvline(x=float(onset), dashes=(5, 1), color='k', lw=1)

  # best onset mark
  if bon_pos is not None:
    best_onset_times = frames_to_time(np.array([bon_pos, bon_pos+frame_size]), fs, hop)
    for onset in best_onset_times:
      plt.axvline(x=float(onset), dashes=(3, 3), color='b', lw=2)

  ax.grid()
  ax.set_title('time signal of ' + '"' + name + '"')
  ax.set_ylabel("magnitude")
  ax.set_xlim([0, t[-1]])

  # select mfcc coeffs in arrays
  if mfcc.shape[0] == 39:
    sel_coefs = [np.arange(0, 12), np.arange(12, 24), np.arange(24, 36), np.arange(36, 39)]
    titles = ['12 MFCCs', 'deltas', 'double deltas', 'energies']
  else:
    sel_coefs = [np.arange(0, mfcc.shape[0])]
    titles = ['MFCCs']

  # mfcc plots
  for i, c in enumerate(sel_coefs):

    # row start and stop
    rs = (i+1) * n_im_rows + 2
    re = (i+2) * n_im_rows

    # specify grid pos
    ax = fig.add_subplot(gs[rs:re, :n_cols-2])

    # plot image coeffs
    im = ax.imshow(mfcc[c], aspect='auto', extent=[0, t[-1], c[-1], c[0]])

    # annotation
    plot_textGrid_annotation(anno_file)

    # some labels
    ax.set_title(titles[i])
    ax.set_ylabel("cepstrum coeff")
    if i == len(sel_coefs) - 1:
      ax.set_xlabel("time [s]")
    ax.set_xlim(left=0)

    # add colorbar
    ax = fig.add_subplot(gs[rs:re, n_cols-1])
    fig.colorbar(im, cax=ax)

  # plot
  if plot_path is not None:
    plt.savefig(plot_path + name + '.png', dpi=150)
  plt.close()


def plot_mfcc_only(mfcc, fs=16000, hop=160, plot_path=None, name='mfcc_only', show_plot=False):
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
  if mfcc.shape[0] == 39:
    sel_coefs = [np.arange(0, 12), np.arange(12, 24), np.arange(24, 36), np.arange(36, 39)]
    titles = ['12 MFCCs' + ' of "' + name + '"', 'deltas', 'double deltas', 'energies']
  else:
    sel_coefs = [np.arange(0, mfcc.shape[0])]
    titles = ['MFCCs' + ' of "' + name + '"',]

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
    plt.savefig(plot_path + name + '.png', dpi=150)
    plt.close()

  elif show_plot:
    plt.show()


def plot_mel_band_weights(w_f, w_mel, f, m, plot_path=None, name='mel_bands', show_plot=False):
  """
  mel band weights
  """

  # plot f bands
  plt.figure(figsize=(8, 5))
  plt.plot(f, w_f.T)
  plt.ylabel('magnitude')
  plt.xlabel('frequency [Hz]')
  plt.grid()

  # plot
  if plot_path is not None:
    plt.savefig(plot_path + name + '_f' + '.png', dpi=150)
    plt.close()

  # plot mel bands
  plt.figure(figsize=(8, 5))
  plt.plot(m, w_mel.T)
  plt.ylabel('magnitude')
  plt.xlabel('mel [mel]')
  plt.grid()

  # plot
  if plot_path is not None:
    plt.savefig(plot_path + name + '_mel' '.png', dpi=150)
    plt.close()

  elif show_plot:
    plt.show()


if __name__ == '__main__':
  
  # metric path
  metric_path = './ignore/plots/ml/metrics/'

  # wav re
  metrics_re = '*.npz'

  # get wavs
  metric_files = glob(metric_path + metrics_re)

  # safety
  if not any(metric_files):
    print("No files available")

  # do something
  else:

    # load files
    data = [np.load(file) for file in metric_files]

    train_loss = data[0]['train_loss'] 
    val_loss = data[0]['val_loss'] 
    val_acc = data[0]['val_acc'] 

    plot_train_loss(train_loss, val_loss, plot_path=None, name='None')
    plot_val_acc(val_acc, plot_path=None, name='None')

    plt.show()