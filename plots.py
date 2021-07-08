"""
plot some figures
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import confusion_matrix
from feature_extraction import onsets_to_onset_times, frames_to_time
from glob import glob
from praatio import tgio

# color palettes
from palettable.cubehelix import red_16
from palettable.cubehelix import purple_16
from palettable.cubehelix import jim_special_16
from palettable.scientific.diverging import Cork_20
from palettable.cartocolors.qualitative import Antique_4
from palettable.cartocolors.qualitative import Prism_8


def get_figsize(context='none'):
  """
  fig sizes 
  """

  # squares
  if context == 'square_big': return (8, 8)
  elif context == 'square_small': return (4, 4)

  # special
  elif context == 'score': return (8, 5)
  elif context == 'mel': return (8, 5)
  elif context == 'half': return (8, 4)
  elif context == 'waveform': return (8, 3)
  elif context == 'shift': return (8, 1.5)

  return (6, 6)

def get_fontsize(context='none', add_size=0):
  """
  get font size
  """

  # init font size
  font_size = 8

  # usuals
  if context == 'title': font_size = 14
  elif context == 'axis_label': font_size = 11
  elif context == 'axis_tick_major': font_size = 10
  elif context == 'axis_tick_minor': font_size = 9

  # special
  elif context == 'colorbar': font_size = 9
  elif context == 'conf_normal': font_size = 10
  elif context == 'conf_small': font_size = 7
  elif context == 'anno': font_size = 12

  # modification
  font_size += add_size

  return font_size


def get_colormap_from_context(context='none'):
  """
  my colormaps
  """

  #reds4 = [red_16.mpl_colors[2], red_16.mpl_colors[5], red_16.mpl_colors[8], red_16.mpl_colors[11]]

  if context == 'mfcc': return red_16.mpl_colormap
  elif context == 'spectogram-log': return red_16.mpl_colormap.reversed()
  #elif context == 'spectogram-log': return red_16.mpl_colormap
  elif context == 'spectogram': return red_16.mpl_colormap.reversed()

  elif context == 'confusion': return ListedColormap(purple_16.mpl_colormap.reversed()(np.linspace(0, 0.8, 25)), name='purpletonian')
  elif context == 'dct-div': return ListedColormap(np.vstack((purple_16.mpl_colormap(np.linspace(0.2, 1, 64)), red_16.mpl_colormap.reversed()(np.linspace(0, 0.8, 64)))), name='rp-div')
  elif context == 'dct': return purple_16.mpl_colormap.reversed()
  #elif context == 'dct': return red_16.mpl_colormap
  #elif context == 'dct': return purple_16.mpl_colormap

  #elif context == 'weight0': return jim_special_16.mpl_colormap
  elif context == 'weight0': return red_16.mpl_colormap
  elif context == 'weight0-div': return ListedColormap(np.vstack((purple_16.mpl_colormap(np.linspace(0, 1, 16)), red_16.mpl_colormap.reversed()(np.linspace(0, 0.8, 16)))), name='rp-div')
  
  #elif context == 'weight1': return Cork_20.mpl_colormap
  elif context == 'weight1': return red_16.mpl_colormap
  elif context == 'weight1-div': return ListedColormap(np.vstack((purple_16.mpl_colormap(np.linspace(0, 1, 16)), red_16.mpl_colormap.reversed()(np.linspace(0, 0.8, 16)))), name='rp-div')
  
  # colors
  elif context == 'wav': return Antique_4.mpl_colors
  elif context == 'wav-hline': return Antique_4.mpl_colors[1]
  elif context == 'hist': return Antique_4.mpl_colors[1]
  #elif context == 'wav-hline': return red_16.mpl_colors[10]
  #elif context == 'loss': return [Prism_8.mpl_colors[2]] + [Prism_8.mpl_colors[5]]
  #elif context == 'loss': return [Prism_8.mpl_colors[2]] + [Prism_8.mpl_colors[5]]
  elif context == 'loss': return [red_16.mpl_colors[5]] + [red_16.mpl_colors[10]]
  #elif context == 'adv-loss': return [red_16.mpl_colors[2], red_16.mpl_colors[5], red_16.mpl_colors[8], red_16.mpl_colors[11]]
  elif context == 'adv-loss': return [red_16.mpl_colors[3], red_16.mpl_colors[5], red_16.mpl_colors[7], red_16.mpl_colors[10]]
  #elif context == 'adv-loss': return Prism_8.mpl_colors[:2] + Prism_8.mpl_colors[6:]
  elif context == 'acc': return Prism_8.mpl_colors[5:]
  #elif context == 'acc': return Prism_8.mpl_colors[5:]
  elif context == 'mel': return Antique_4.mpl_colors[3:]

  elif context == 'bench-noise': return ListedColormap(red_16.mpl_colormap.reversed()(np.linspace(0, 0.6, 10)), name='red_short')
  elif context == 'bench-noise-2': return ListedColormap(red_16.mpl_colormap.reversed()(np.linspace(0, 0.6, 2)), name='red_short')
  #elif context == 'bench-noise': return ListedColormap(purple_16.mpl_colormap.reversed()(np.linspace(0, 0.6, 10)), name='purple_short')
  
  elif context == 'bench-shift': return ListedColormap(red_16.mpl_colormap.reversed()(np.linspace(0, 0.6, 10)), name='red_short')
  elif context == 'bench-shift-2': return ListedColormap(red_16.mpl_colormap.reversed()(np.linspace(0, 0.6, 2)), name='red_short')
  #elif context == 'bench-shift': return ListedColormap(np.vstack((red_16.mpl_colormap.reversed()(np.linspace(0, 0, 5)), red_16.mpl_colormap.reversed()(np.linspace(0.2, 0.6, 5)))), name='red_short')
  #elif context == 'bench': return purple_16.mpl_colormap.reversed()
  #elif context == 'bench': return purple_16.mpl_colormap.reversed()

  return None


def add_colorbar(fig, im, cax=None, size='2%', pad='2%'):
  """
  adds colorbar to plot
  """

  # devider for cax
  if cax is None: cax = make_axes_locatable(plt.gca()).append_axes('right', size=size, pad=pad)

  # colorbar
  color_bar = fig.colorbar(im, cax=cax)
  color_bar.ax.tick_params(labelsize=get_fontsize('colorbar'))


def plot_mfcc_anim(x, cmap=None, plot_path=None, name='mfcc-anim'):
  """
  mfcc animation (x is list of images)
  """

  # get cmap
  if cmap is None: cmap = get_colormap_from_context(context='mfcc')

  # plot
  fig = plt.figure(figsize=get_figsize(context='square_big'))
  plt.axis("off")

  # animation
  ani = animation.ArtistAnimation(fig, [[plt.imshow(np.squeeze(i)[0, :], interpolation='none', cmap=cmap, animated=True)] for i in x], interval=1000, repeat_delay=1000, blit=True)

  # plot save and show
  if plot_path is not None: ani.save(plot_path + name + '.mp4')
    
  # show
  plt.show()


def plot_histogram(x, bins=None, color=None, y_log_scale=False, x_log_scale=False, x_label='', y_label='', title='', plot_path=None, name='hist', show_plot=False):
  """
  histogram plot
  """

  # get cmap
  if color is None: color = get_colormap_from_context(context='hist')

  # init figure
  fig = plt.figure(figsize=get_figsize(context='square_big'))

  # plot hist
  im = plt.hist(x, bins=bins, color=color)

  # layout
  plt.title(title, fontsize=get_fontsize('title')), plt.xlabel(x_label, fontsize=get_fontsize('axis_label')), plt.ylabel(y_label, fontsize=get_fontsize('axis_label'))
  plt.grid()

  # log scaling
  if y_log_scale: plt.yscale('log')
  if x_log_scale: plt.xscale('log')

  # tight plot
  plt.tight_layout()

  # plot save and show
  if plot_path is not None: plt.savefig(plot_path + name + '.png', dpi=150)
  
  # show plot
  if show_plot: plt.show()

  return fig


def plot_test_bench_noise(x, y, snrs, cmap=None, context='bench-noise', title='noise', plot_path=None, name='test_bench_noise', show_plot=False):
  """
  shiftinvariant test
  """

  # get cmap
  if cmap is None: cmap = get_colormap_from_context(context=context)

  # plot init
  fig = plt.figure(figsize=get_figsize(context='half'))

  # image
  ax = plt.axes()
  im = ax.pcolormesh(x, edgecolors='k', linewidth=1, vmax=1, vmin=0, cmap=cmap)

  # design
  plt.title(title)
  plt.xlabel("SNR [dB]")

  # tick adjustment
  ax.set_yticks(np.arange(0.5, len(y), 1))
  ax.set_yticklabels(y, fontsize=get_fontsize('axis_label'))

  ax.set_xticks(np.arange(0.5, len(snrs), 1))
  ax.set_xticklabels(snrs, fontsize=get_fontsize('axis_label'))

  # aspect
  ax.set_aspect('equal')

  # colorbar
  add_colorbar(fig, im)

  # tight plot
  plt.tight_layout()

  # plot save and show
  if plot_path is not None: 
    plt.savefig(plot_path + name + '.png', dpi=150)
    plt.close()
    
  if show_plot: plt.show()

  return fig


def plot_test_bench_shift(x, y, cmap=None, context='bench-shift', title='shift', plot_path=None, name='test_bench_shift', show_plot=False):
  """
  shiftinvariant test
  """

  # get cmap
  if cmap is None: cmap = get_colormap_from_context(context=context)

  # plot init
  fig = plt.figure(figsize=get_figsize(context='shift'))

  # image
  ax = plt.axes()
  #im = ax.imshow(x, aspect='equal', interpolation='none')

  im = ax.pcolormesh(x, edgecolors='k', linewidth=1, vmax=1, vmin=0, cmap=cmap)

  # design
  plt.title(title)
  plt.xlabel("shift index")

  # tick adjustment
  ax.set_yticks(np.arange(0.5, len(y), 1))
  ax.set_yticklabels(y, fontsize=get_fontsize('axis_label'))

  ax.tick_params(axis='x', which='major', labelsize=get_fontsize('axis_tick_major'))

  # aspect
  ax.set_aspect('equal')

  # colorbar
  add_colorbar(fig, im)

  # tight plot
  plt.tight_layout()

  # plot save and show
  if plot_path is not None: 
    plt.savefig(plot_path + name + '.png', dpi=150)
    plt.close()
    
  if show_plot: plt.show()

  return fig


def plot_wav_grid(wav_info_dicts, feature_params, grid_size=(8, 8), cmap=None, title='', plot_path=None, name='wav_grid', show_plot=False):
  """
  plot a wav grid 
  """

  # shortcuts
  fs, hop = feature_params['fs'], int(feature_params['hop_s'] * feature_params['fs'])

  # time vector
  t = np.arange(0, len(wav_info_dicts[0]['x'])/fs, 1/fs)

  # get cmap
  if cmap is None: cmap = get_colormap_from_context(context='wav')

  # make a grid
  n_im_rows, n_im_cols, r_space, c_space = 3, 5, 1, 1
  n_rows, n_cols = n_im_rows * grid_size[1] + r_space * grid_size[1] - 1, n_im_cols * grid_size[0] + c_space * grid_size[0] - 1

  # init figure
  fig, gs = plt.figure(figsize=get_figsize(context='square_big')), plt.GridSpec(n_rows, n_cols, wspace=0.4, hspace=0.3)

  # overall layout
  plt.title(title), plt.axis('off')

  # init indices
  i, j = 0, 0

  # mfcc plots
  for wav_info_dict in wav_info_dicts:

    # row start and stop
    rs, re = j * n_im_rows + j * r_space, (j + 1) * n_im_rows + j * r_space
    cs, ce = i * n_im_cols + i * c_space, (i + 1) * n_im_cols + i * c_space
    #print("rs: {}, re: {}, cs: {}, ce: {}".format(rs, re, cs, ce))

    # exception for one row images
    if rs == re: re += 1

    # update indices
    if not i % (grid_size[0]-1) and i: i, j = 0, j + 1
    else: i += 1

    # specify grid pos
    ax = fig.add_subplot(gs[rs:re, cs:ce])

    # wav color
    if cmap is not None: ax.set_prop_cycle('color', cmap)

    # plot
    ax.plot(t[:len(wav_info_dict['x'])], wav_info_dict['x'])

    # best onset mark
    if wav_info_dict['bon_pos'] is not None:
      for onset in np.array([wav_info_dict['bon_pos'] / fs, wav_info_dict['bon_pos'] / fs + feature_params['frame_size_s']]): plt.axvline(x=float(onset), dashes=(3, 2), color=get_colormap_from_context(context='wav-hline'), lw=2)

    # layout
    ax.set_ylim([-1, 1]), ax.axis('off'), ax.set_title(wav_info_dict['y'], fontsize=get_fontsize('title'))

  # tight plot
  plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.97, wspace=0, hspace=0)

  # plot save and show
  if plot_path is not None: plt.savefig(plot_path + name + '.png', dpi=150)

  # show plot
  if show_plot: plt.show()


def plot_other_grid(x, grid_size=(8, 8), title='grid1', plot_path=None, name='grid1', show_plot=False):
  """
  plot grid
  """

  # make a grid
  n_im_rows, n_im_cols = x.shape[2], x.shape[3]
  n_rows, n_cols = n_im_rows * grid_size[1] + grid_size[1]+1, n_im_cols * grid_size[0] + grid_size[0]+1

  fig = plt.figure(figsize=(8*n_cols/n_rows-0.5, 8))
  gs = plt.GridSpec(n_rows, n_cols, wspace=0.4, hspace=0.3)
  plt.title(title)
  plt.axis("off")

  # init indices
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

  # plot save and show
  if plot_path is not None: plt.savefig(plot_path + name + '.png', dpi=150)

  # show plot
  if show_plot: plt.show()


def plot_grid_images(x, padding=1, num_cols=8, cmap=None, context='none', color_balance=False, title='grid2', plot_path=None, name='grid2', show_plot=False):
  """
  plot grid images
  """

  # get cmap
  if cmap is None: cmap = get_colormap_from_context(context=context)

  # cast to numpy
  x = np.array(x)

  # max value
  vmax = np.max(np.abs(x)) if color_balance else None
  vmin = -vmax if color_balance else None

  # extend for wavenet
  if len(x.shape) != 4: x = x[:, :, np.newaxis, :]

  # get dimensions
  n_kernels, n_channels, n_features, n_frames = x.shape

  # determine minimum value
  value_min = vmin if color_balance else np.min(x)

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

  # get all channels together
  for ch_grid_img in grid_img:
    all_grid_img = np.concatenate((all_grid_img, ch_grid_img), axis=0)

  # last padding
  all_grid_img = np.concatenate((all_grid_img, h_padding[0]), axis=0)


  # plot init
  m, n = all_grid_img.shape
  fig = plt.figure(figsize=(8, np.clip(m / n * 8, 1, 8))) if m < n else plt.figure(figsize=(np.clip(n / m * 8, 2, 8), 8))
  #if m < n: fig = plt.figure(figsize=(8, np.clip(m / n * 8, 1, 8)))
  #else: fig = plt.figure(figsize=(np.clip(n / m * 8, 2, 8), 8))

  # image
  ax = plt.axes()
  im = ax.imshow(all_grid_img, aspect='equal', interpolation='none', cmap=cmap, vmax=vmax, vmin=vmin)

  # design
  plt.axis("off")
  plt.title(title, fontsize=get_fontsize('title'))

  # colorbar
  add_colorbar(fig, im)

  # tight plot
  plt.subplots_adjust(left=0.10, bottom=0.02, right=0.90, top=0.90, wspace=0, hspace=0)
  #plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.97, wspace=0, hspace=0)

  # plot save and show
  if plot_path is not None: 
    plt.savefig(plot_path + name + '.png', dpi=150)
    plt.close()
    
  if show_plot: plt.show()

  return fig


def plot_damaged_file_score(z, plot_path=None, name='z_score', show_plot=False):
  """
  damaged file score
  """

  # setup figure
  fig = plt.figure(figsize=get_figsize('half'))
  plt.scatter(x=z, y=np.zeros(len(z)))
  plt.ylabel('nothing')
  plt.xlabel('score')
  plt.grid()

  # plot the fig
  if plot_path is not None: plt.savefig(plot_path + name + '.png', dpi=150)

  # show plot
  if show_plot: plt.show()


def plot_waveform(x, fs, e=None, anno_file=None, bon_samples=None, hop=None, onset_frames=None, y_ax_balance=True, cmap=None, title='', xlim=None, ylim=None, plot_path=None, name='None', show_plot=False, close_plot=False):
  """
  just a simple waveform
  """

  # time vector
  t = np.arange(0, len(x)/fs, 1/fs)

  # setup figure
  fig = plt.figure(figsize=get_figsize('waveform'))

  # create axis
  ax = plt.axes()

  # get cmap
  if cmap is None: cmap = get_colormap_from_context(context='wav')
  if cmap is not None: ax.set_prop_cycle('color', cmap)

  # plot signal
  ax.plot(t, x)

  # energy plot
  if e is not None: ax.plot(np.arange(0, len(x)/fs, 1/fs * hop), e)

  # draw onsets
  if onset_frames is not None:
    for onset in frames_to_time(onset_frames, fs, hop): plt.axvline(x=float(onset), dashes=(5, 1), color='k')

  if bon_samples is not None:
    for bon_sample in bon_samples: plt.axvline(x=float(bon_sample)/fs, dashes=(5, 1), color=get_colormap_from_context(context='wav-hline'))

  # lims
  if xlim is not None: plt.xlim(xlim)
  if ylim is not None: plt.ylim(ylim)

  if y_ax_balance:
    v = np.max(np.abs(x))
    ax.set_xlim([0, 1])
    ax.set_ylim([-v - 0.1 * v, v + 0.1 * v])

  # annotation
  if anno_file is not None: plot_textGrid_annotation(anno_file, x=x, plot_text=True)

  # care about labels
  ax.set_xticks(t[::1600])
  ax.set_xticklabels(['{:.1f}'.format(ti) for ti in t[::1600]])
  ax.tick_params(axis='both', which='major', labelsize=get_fontsize('axis_tick_major')), ax.tick_params(axis='both', which='minor', labelsize=get_fontsize('axis_tick_minor'))

  # title
  if len(title): plt.title(title, fontsize=get_fontsize('title'))

  # layout
  plt.ylabel('magnitude', fontsize=get_fontsize('axis_label')), plt.xlabel('time [s]', fontsize=get_fontsize('axis_label')), plt.grid()

  # tight plot
  plt.tight_layout()

  # plot, show and close
  if plot_path is not None: plt.savefig(plot_path + name + '.png', dpi=150)
  if show_plot: plt.show()
  if close_plot: plt.close()

  return fig


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


def plot_confusion_matrix(cm, classes, cmap=None, plot_path=None, name='None'):
  """
  plot confusion matrix
  """

  # return if empty
  if cm is None: return

  # get cmap
  if cmap is None: cmap = get_colormap_from_context(context='confusion')

  # init plot
  fig = plt.figure(figsize=get_figsize(context='square_big'))

  # axis
  ax = plt.axes()

  # image
  im = ax.imshow(cm, cmap=cmap, interpolation='none', vmin=0, vmax=np.max(np.sum(cm, axis=1)))

  # text handling
  for y_pred_pos in range(len(classes)):
    for y_true_pos in range(len(classes)):

      # font color and size
      font_color = 'black' if y_pred_pos != y_true_pos else 'white'
      fontsize = get_fontsize('conf_small') if len(classes) > 10 else get_fontsize('conf_normal')

      # write numbers inside
      text = ax.text(y_true_pos, y_pred_pos, cm[y_pred_pos, y_true_pos], ha='center', va='center', color=font_color, fontsize=fontsize)

  # care about labels
  ax.set_xticks(np.arange(len(classes))), ax.set_yticks(np.arange(len(classes)))
  ax.set_xticklabels(classes), ax.set_yticklabels(classes)

  plt.xticks(fontsize=get_fontsize('axis_tick_major'), rotation=90), plt.yticks(fontsize=get_fontsize('axis_tick_major'), rotation=0)
  plt.xlabel('predicted labels'), plt.ylabel('true labels')

  # colorbar
  add_colorbar(fig, im)

  # tight plot
  plt.tight_layout()

  # plot the fig
  if plot_path is not None:
    plt.savefig(plot_path + name + '.png', dpi=150)
    plt.close()


def plot_train_score(train_score, plot_path, name_ext=''):
  """
  plot train scores
  """

  # adversarial nets
  if train_score.__class__.__name__ == 'AdversarialTrainScore':
    plot_adv_train_loss(train_score, plot_path=plot_path, name='score_loss' + name_ext)

  # wave nets
  elif train_score.__class__.__name__ == 'WavenetTrainScore':
    plot_wavenet_train_loss(train_score, plot_path=plot_path, name='score_loss' + name_ext)
    plot_val_acc(train_score.score_dict['val_acc'], plot_path=plot_path, name='score_val_acc' + name_ext)

  # other nets
  else:
    plot_train_loss(train_score.score_dict['train_loss'], train_score.score_dict['val_loss'], plot_path=plot_path, name='train_loss' + name_ext)
    plot_val_acc(train_score.score_dict['val_acc'], plot_path=plot_path, name='score_val_acc' + name_ext)
    

def plot_wavenet_train_loss(train_score, cmap=None, plot_path=None, name='score_loss', show_plot=False):
  """
  wavenet train loss
  """

  # get cmap
  if cmap is None: cmap = get_colormap_from_context(context='adv-loss')

  # setup figure
  fig = plt.figure(figsize=get_figsize(context='score'))

  # create axis
  ax = plt.axes()
  if cmap is not None: ax.set_prop_cycle('color', cmap)

  # plot scores
  ax.plot(train_score.score_dict['loss_t'], label='wave loss')
  ax.plot(train_score.score_dict['loss_y'], label='class loss')

  # layout
  plt.ylabel("loss"), plt.xlabel("iterations"), plt.legend(), plt.grid()

  # tight plot
  plt.tight_layout()

  # plot the fig
  if plot_path is not None:
    plt.savefig(plot_path + name + '.png', dpi=150)
    plt.close()

  # show plot
  if show_plot: plt.show()
  return fig


def plot_adv_train_loss(train_score, cmap=None, plot_path=None, name='score_loss', show_plot=False):
  """
  adversarial train loss
  """

  # get cmap
  if cmap is None: cmap = get_colormap_from_context(context='adv-loss')

  # setup figure
  fig = plt.figure(figsize=get_figsize(context='score'))

  # create axis
  ax = plt.axes()
  if cmap is not None: ax.set_prop_cycle('color', cmap)

  # plots
  ax.plot(train_score.score_dict['g_loss_fake'], label='g_loss_fake')
  ax.plot(train_score.score_dict['g_loss_sim'], label='g_loss_sim')
  ax.plot(train_score.score_dict['d_loss_fake'], label='d_loss_fake')
  ax.plot(train_score.score_dict['d_loss_real'], label='d_loss_real')

  # layout
  plt.ylabel("loss"), plt.xlabel("iterations"), plt.legend(), plt.grid()

  # tight plot
  plt.tight_layout()

  # plot the fig
  if plot_path is not None:
    plt.savefig(plot_path + name + '.png', dpi=150)
    plt.close()

  # show plot
  if show_plot: plt.show()
  return fig


def plot_train_loss(train_loss, val_loss, cmap=None, plot_path=None, name='score_loss', show_plot=False):
  """
  plot train vs. validation loss
  """

  # get cmap
  if cmap is None: cmap = get_colormap_from_context(context='loss')

  # normalize
  if np.linalg.norm(train_loss, ord=np.infty): train_loss = train_loss / np.linalg.norm(train_loss, ord=np.infty)
  if np.linalg.norm(val_loss, ord=np.infty): val_loss = val_loss / np.linalg.norm(val_loss, ord=np.infty)

  # setup figure
  fig = plt.figure(figsize=get_figsize(context='score'))

  # create axis
  ax = plt.axes()
  if cmap is not None: ax.set_prop_cycle('color', cmap)

  # plot
  ax.plot(train_loss, label='train loss'), ax.plot(val_loss, label='val loss')

  # layout
  plt.ylabel("normalized loss"), plt.xlabel("iterations"), plt.legend(), plt.grid()

  # tight plot
  plt.tight_layout()

  # plot the fig
  if plot_path is not None:
    plt.savefig(plot_path + name + '.png', dpi=150)
    plt.close()

  # show plot
  if show_plot: plt.show()

  return fig


def plot_val_acc(val_acc, cmap=None, plot_path=None, name='score_val', show_plot=False):
  """
  plot train vs. validation loss
  """

  # get cmap
  if cmap is None: cmap = get_colormap_from_context(context='acc')

  # setup figure
  fig = plt.figure(figsize=get_figsize(context='score'))

  # create axis
  ax = plt.axes()
  if cmap is not None: ax.set_prop_cycle('color', cmap)

  # plot
  ax.plot(val_acc, label='val acc')

  # ticks
  ax.yaxis.set_major_locator(MultipleLocator(10))

  # layout
  plt.ylabel("accuracy"), plt.xlabel("iterations"), plt.ylim(0, 100), plt.legend(), plt.grid()

  # tight plot
  plt.tight_layout()

  # plot the fig
  if plot_path is not None:
    plt.savefig(plot_path + name + '.png', dpi=150)
    plt.close()

  # show plot
  if show_plot: plt.show()

  return fig


def plot_textGrid_annotation(anno_file, x=None, hop_space=None, plot_text=False, color='k'):
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

      # translate to hop space
      if hop_space is not None: s = s * hop_space - 0.5

      # plot line
      plt.axvline(x=s, dashes=(3, 3), color=color, lw=1)

      # plot text
      if plot_text: plt.text(s + 0.01, h, l, color=color, fontsize=get_fontsize('anno'))


def plot_mfcc_profile(x, fs, N, hop, mfcc, sep_features=True, diff_plot=False, cmap=None, cmap_wav=None, anno_file=None, onsets=None, bon_pos=None, mient=None, minreg=None, frame_size=32, plot_path=None, name='mfcc_profile', close_plot=True, show_plot=False):
  """
  plot mfcc extracted features from audio file
  """

  # to image data
  if len(mfcc.shape) == 3: mfcc = np.squeeze(mfcc.reshape(1, -1, mfcc.shape[2]))

  # get cmap
  if cmap is None: cmap = get_colormap_from_context(context='mfcc')
  if cmap_wav is None: cmap_wav = get_colormap_from_context(context='wav')

  # time vectors
  t, t_hop = np.arange(0, len(x)/fs, 1/fs), np.arange(0, mfcc.shape[1] * hop / fs, hop/fs)

  # s, 1, c, d, d, e
  grid_size = (6, 1) if mfcc.shape[0] == 39 and sep_features else (2, 1)
  grid_row_usage = (0.1, 0.1, 0.2, 0.2, 0.2, 0.2) if mfcc.shape == 39 and sep_features else (0.4, 0.6)

  # make a grid
  n_im_rows, n_im_cols, r_space, c_space = (20, 103, 25, 1) if mfcc.shape[0] == 39 and sep_features else (20, 103, 10, 1)

  # specify number of rows and cols
  n_rows, n_cols = n_im_rows * grid_size[0] + r_space * grid_size[0] - 1, n_im_cols * grid_size[1] + c_space * grid_size[0] - 1

  # init figure
  fig = plt.figure(figsize=get_figsize(context='square_big')) if mfcc.shape[0] == 39 and sep_features else plt.figure(figsize=get_figsize(context='half'))

  # create grid
  gs = plt.GridSpec(n_rows, n_cols, wspace=0.4, hspace=0.3)

  # time series plot
  #ax = fig.add_subplot(gs[0:n_im_rows-1, :n_im_cols-3])
  ax = fig.add_subplot(gs[0:int(n_im_rows * grid_row_usage[0]), :n_im_cols-3])

  # wav
  if cmap_wav is not None: ax.set_prop_cycle('color', cmap_wav)
  ax.plot(t[:len(x)], x)

  # care about labels
  ax.set_xticks(t[::1600])
  ax.set_xticklabels(['{:.1f}'.format(ti) for ti in t[::1600]])
  ax.tick_params(axis='both', which='major', labelsize=get_fontsize('axis_tick_major')), ax.tick_params(axis='both', which='minor', labelsize=get_fontsize('axis_tick_minor'))

  # annotation
  plot_textGrid_annotation(anno_file, x, plot_text=True)

  # min energy time and region
  if mient is not None: plt.axvline(x=mient, dashes=(5, 5), color='r', lw=2)
  if minreg is not None: plt.axvline(x=minreg, dashes=(5, 5), color='g', lw=2)

  # onset marks
  if onsets is not None:
    onset_times = onsets_to_onset_times(onsets, fs, N, hop)
    for onset in onset_times: plt.axvline(x=float(onset), dashes=(5, 1), color='k', lw=1)

  # best onset mark
  if bon_pos is not None:
    for onset in frames_to_time(np.array([bon_pos, bon_pos+frame_size]), fs, hop): plt.axvline(x=float(onset), dashes=(3, 2), color=get_colormap_from_context(context='wav-hline'), lw=2)

  # layout of time plot
  #ax.set_title('Time Signal of ' + '"' + name + '"', fontsize=get_fontsize('title')), ax.set_ylabel("magnitude", fontsize=get_fontsize('title'))
  ax.set_title('Time Signal', fontsize=get_fontsize('title')), ax.set_ylabel("magnitude", fontsize=get_fontsize('axis_label'))
  ax.set_xlim([0, t[-1]]), ax.grid()

  # select mfcc coeffs in arrays
  if mfcc.shape[0] == 39 and sep_features: sel_coefs, titles = [[0, 13, 26], np.arange(1, 12), np.arange(14, 25), np.arange(27, 38), [12, 25, 38]], ['1st cepstral coefficient with deltas', 'cepstral coefficients', 'deltas', 'double deltas', 'energies']
  else: sel_coefs, titles = [np.arange(0, mfcc.shape[0])], ['MFCCs']

  # mfcc plots
  for i, c in enumerate(sel_coefs, start=1):

    # row start and stop
    rs, re = int(i * n_im_rows * grid_row_usage[i-1]) + i * r_space, int((i + 1) * n_im_rows * np.sum([grid_row_usage[j] for j in range(i + 1)])) + i * r_space
    
    # specify grid pos
    ax = fig.add_subplot(gs[rs:re, :n_im_cols-3-2])

    # plot image coeffs
    im = ax.imshow(mfcc[c], aspect='auto', interpolation='none', cmap=cmap)

    # care about labels
    ax.set_xticks(np.arange(len(t_hop))[::10] - 0.5)
    ax.set_xticklabels(['{:.1f}'.format(ti) for ti in t_hop[::10]])
    if len(c) < 5: ax.set_yticks(np.arange(0, len(c))), ax.set_yticklabels(c , fontsize=get_fontsize('axis_tick_major'))
    if len(c) > 15: ax.set_yticks(np.arange(0, len(c))[::5]), ax.set_yticklabels(c[::5], fontsize=get_fontsize('axis_tick_major'))   
    else: ax.set_yticks(np.arange(0, len(c))[::3]), ax.set_yticklabels(c[::3], fontsize=get_fontsize('axis_tick_major'))    
    ax.tick_params(axis='both', which='major', labelsize=get_fontsize('axis_tick_major')), ax.tick_params(axis='both', which='minor', labelsize=get_fontsize('axis_tick_minor'))

    # annotation
    plot_textGrid_annotation(anno_file, hop_space=fs/hop)

    # some labels
    ax.set_title(titles[i-1], fontsize=get_fontsize('title')), ax.set_ylabel("mfcc coeff.", fontsize=get_fontsize('axis_label'))
    if i == len(sel_coefs): ax.set_xlabel("time [s]", fontsize=get_fontsize('axis_label'))

    # add colorbar
    ax = fig.add_subplot(gs[rs:re, n_im_cols-3:n_im_cols-1])
    add_colorbar(fig, im, cax=ax)

  if diff_plot and mfcc.shape[0] != 39:
    rs, re = n_im_rows * (1 + len(sel_coefs)) + 1, -2
    ax = fig.add_subplot(gs[rs:re, :n_im_cols-5])
    im = ax.imshow(np.diff(mfcc, axis=1), aspect='auto', interpolation='none', cmap=cmap)
    #im = ax.imshow(np.array([np.sum(mfcc**2, axis=0) / np.max(np.sum(mfcc**2, axis=0))]), aspect='auto', interpolation='none', cmap=cmap)

    # add colorbar
    ax = fig.add_subplot(gs[rs:re, n_im_cols-3:n_im_cols-1])
    add_colorbar(fig, im, cax=ax)


  # tight plot
  plt.subplots_adjust(left=0.1, bottom=0.00, right=0.97, top=0.93, wspace=0, hspace=0) if mfcc.shape[0] == 39 and sep_features else plt.subplots_adjust(left=0.1, bottom=0.00, right=0.94, top=0.90, wspace=0, hspace=0) 

  # save
  if plot_path is not None: plt.savefig(plot_path + name + '.png', dpi=150)

  # close plot
  if close_plot: plt.close()

  # show plot
  if show_plot: plt.show()


def plot_mfcc_only(mfcc, fs=16000, hop=160, cmap=None, context='mfcc', plot_path=None, name='mfcc_only', show_plot=False):
  """
  plot mfcc extracted features only (no time series)
  mfcc: [m x l]
  """

  # get cmap
  if cmap is None: cmap = get_colormap_from_context(context=context)

  # get shape
  m, l = mfcc.shape

  # time vector
  t = np.arange(0, l * hop / fs, hop/fs)

  # setup figure
  fig = plt.figure(figsize=get_figsize(context='half'))

  # select mfcc coeffs in arrays
  if mfcc.shape[0] == 39:
    sel_coefs = [np.arange(0, 12), np.arange(12, 24), np.arange(24, 36), np.arange(36, 39)]
    titles = ['12 MFCCs' + ' of "' + name + '"', 'deltas', 'double deltas', 'energies']
    n_rows, n_cols, n_im_rows = 20, 20, 15

  else:
    sel_coefs = [np.arange(0, mfcc.shape[0])]
    titles = ['MFCCs' + ' of "' + name + '"',]
    n_rows, n_cols, n_im_rows = 20, 20, 8

  # grid
  gs = plt.GridSpec(n_rows, n_cols, wspace=0.4, hspace=0.3)

  # mfcc plots
  for i, c in enumerate(sel_coefs):

    # row start and stop
    rs = (i) * n_im_rows + 2
    re = (i+1) * n_im_rows

    # specify grid pos
    ax = fig.add_subplot(gs[rs:re, :n_cols-2])

    # plot selected mfcc
    im = ax.imshow(mfcc[c], aspect='auto', extent=[0, t[-1], c[-1], c[0]], cmap=cmap)
    #im = ax.imshow(mfcc[c], aspect='equal', interpolation='none', extent=[0, t[-1], c[-1], c[0]], cmap=cmap)

    # some labels
    ax.set_title(titles[i])
    ax.set_ylabel("cepstrum coeff")
    if i == len(sel_coefs) - 1: ax.set_xlabel("time [s]")
    ax.set_xlim(left=0)

    # add colorbar
    ax = fig.add_subplot(gs[rs:re, n_cols-1])
    add_colorbar(fig, im, cax=ax)

  # plot the fig
  if plot_path is not None:
    plt.savefig(plot_path + name + '.png', dpi=150)
    plt.close()

  elif show_plot:
    plt.show()


def plot_mfcc_equal_aspect(mfcc, fs=16000, hop=160, cmap=None, context='mfcc', plot_path=None, name='mfcc_equal', gizmos_off=False, show_plot=False):
  """
  plot mfcc extracted features with equal aspect
  """

  # get cmap
  if cmap is None: cmap = get_colormap_from_context(context=context)

  # get shape
  m, l = mfcc.shape

  # time vector
  t = np.arange(0, l * hop / fs, hop/fs)

  # setup figure
  fig = plt.figure(figsize=(5 * l / m / 2, 5))

  # axis and plot
  ax = plt.axes()
  im = ax.imshow(mfcc, aspect='equal', interpolation='none', cmap=cmap)

  # axis off
  if gizmos_off: plt.axis("off")

  if not gizmos_off:

    # care about labels
    ax.set_xticks(np.arange(len(t))[::5])
    ax.set_xticklabels(['{:.2f}'.format(ti) for ti in t[::5]])

    # colorbar
    add_colorbar(fig, im)

    # some labels
    ax.set_title('MFCCs' + ' of "' + name + '"')
    ax.set_ylabel("cepstrum coeff"), ax.set_xlabel("time [s]")

  # tight plot
  plt.tight_layout()

  # plot the fig
  if plot_path is not None:
    plt.savefig(plot_path + name + '.png', dpi=150)
    plt.close()

  elif show_plot:
    plt.show()


def plot_mel_band_weights(w_f, w_mel, f, m, cmap=None, plot_path=None, name='mel_bands', show_plot=False):
  """
  mel band weights
  """

  # get cmap
  if cmap is None: cmap = get_colormap_from_context(context='adv-loss')

  # plot f bands
  fig1 = plt.figure(figsize=get_figsize(context='score'))

  # create axis
  ax = plt.axes()
  if cmap is not None: ax.set_prop_cycle('color', cmap)

  # plot frequency weights
  ax.plot(f, w_f.T)

  # tick size
  ax.tick_params(axis='both', which='major', labelsize=get_fontsize('axis_tick_major', add_size=2)), ax.tick_params(axis='both', which='minor', labelsize=get_fontsize('axis_tick_minor', add_size=2))

  # layout
  plt.ylabel('magnitude', fontsize=get_fontsize('axis_label', add_size=2)), plt.xlabel('frequency [Hz]', fontsize=get_fontsize('axis_label', add_size=2)), plt.grid()

  # tight plot
  plt.tight_layout()

  # save plot
  if plot_path is not None: plt.savefig(plot_path + name + '_f' + '.png', dpi=150)

  # plot mel bands
  fig2 = plt.figure(figsize=get_figsize(context='score'))

  # create axis
  ax = plt.axes()
  if cmap is not None: ax.set_prop_cycle('color', cmap)

  # plot
  ax.plot(m, w_mel.T)

  # tick size
  ax.tick_params(axis='both', which='major', labelsize=get_fontsize('axis_tick_major', add_size=2)), ax.tick_params(axis='both', which='minor', labelsize=get_fontsize('axis_tick_minor', add_size=2))

  # layout
  plt.ylabel('magnitude', fontsize=get_fontsize('axis_label', add_size=2)), plt.xlabel('mel [mel]', fontsize=get_fontsize('axis_label', add_size=2)), plt.grid()

  # tight plot
  plt.tight_layout()

  # save plot
  if plot_path is not None: plt.savefig(plot_path + name + '_mel' '.png', dpi=150)

  # show plot
  if show_plot: plt.show()


def plot_mel_scale(cmap=None, plot_path=None, name='mel', show_plot=False):
  """
  mel band weights
  """

  from feature_extraction import f_to_mel

  # get cmap
  if cmap is None: cmap = get_colormap_from_context(context='mel')

  # frequency
  f = np.arange(0, 16000, 1)

  # mel
  m = f_to_mel(f)

  # plot mel bands
  fig = plt.figure(figsize=get_figsize(context='score'))

  # create axis
  ax = plt.axes()
  if cmap is not None: ax.set_prop_cycle('color', cmap)

  # plot
  ax.plot(f, m)

  # tick size
  ax.tick_params(axis='both', which='major', labelsize=get_fontsize('axis_tick_major', add_size=2)), ax.tick_params(axis='both', which='minor', labelsize=get_fontsize('axis_tick_minor', add_size=2))

  # layout
  plt.ylabel('mel [mel]', fontsize=get_fontsize('axis_label', add_size=2)), plt.xlabel('frequency [Hz]', fontsize=get_fontsize('axis_label', add_size=2)), plt.grid()

  # tight plot
  plt.tight_layout()

  # save plot
  if plot_path is not None: plt.savefig(plot_path + name + '.png', dpi=150)

  # show plot
  if show_plot: plt.show()


def plot_dct(h, cmap=None, context='dct', plot_path=None, name='dct', show_plot=False):
  """
  discrete cosine transform
  """

  # get cmap
  if cmap is None: cmap = get_colormap_from_context(context=context)

  # init
  fig = plt.figure(figsize=get_figsize(context='square_small'))

  # axis
  ax = plt.axes()

  # image
  im = ax.imshow(h, cmap=cmap, interpolation='none', vmin=-1, vmax=1)

  # layout
  plt.axis("off")

  # colorbar
  add_colorbar(fig, im, size='4%', pad='4%')

  # tight plot
  plt.tight_layout()

  # save plot
  if plot_path is not None: plt.savefig(plot_path + name + '.png', dpi=150)
  
  # show plot
  if show_plot: plt.show()


def plot_spec_profile(x, x_spec, fs, N, hop, log_scale=False, cmap=None, cmap_wav=None, anno_file=None, bon_pos=None, frame_size=50, plot_path=None, title='', name='spec', show_plot=True):
  """
  spectogram with waveform
  """

  # get cmap
  if cmap is None: cmap = get_colormap_from_context(context='spectogram') if not log_scale else get_colormap_from_context(context='spectogram-log')
  if cmap_wav is None: cmap_wav = get_colormap_from_context(context='wav')

  # time vectors
  t, t_hop = np.arange(0, len(x)/fs, 1/fs), np.arange(0, x_spec.shape[1] * hop / fs, hop/fs)

  # s, 1, c, d, d, e
  grid_size = (2, 1)
  grid_row_usage = (0.4, 0.6)

  # make a grid
  n_im_rows, n_im_cols, r_space, c_space = (20, 103, 10, 1)

  # specify number of rows and cols
  n_rows, n_cols = n_im_rows * grid_size[0] + r_space * grid_size[0] - 1, n_im_cols * grid_size[1] + c_space * grid_size[0] - 1

  # init figure
  fig = plt.figure(figsize=get_figsize(context='half'))

  # create grid
  gs = plt.GridSpec(n_rows, n_cols, wspace=0.4, hspace=0.3)

  # time series plot
  ax = fig.add_subplot(gs[0:int(n_im_rows * grid_row_usage[0]), :n_im_cols-3])

  # wav
  if cmap_wav is not None: ax.set_prop_cycle('color', cmap_wav)
  ax.plot(t[:len(x)], x)

  # care about labels
  ax.set_xticks(t[::1600])
  ax.set_xticklabels(['{:.1f}'.format(ti) for ti in t[::1600]])
  ax.tick_params(axis='both', which='major', labelsize=get_fontsize('axis_tick_major')), ax.tick_params(axis='both', which='minor', labelsize=get_fontsize('axis_tick_minor'))

  # annotation
  plot_textGrid_annotation(anno_file, x, plot_text=True)

  # best onset mark
  if bon_pos is not None:
    for onset in frames_to_time(np.array([bon_pos, bon_pos+frame_size]), fs, hop): plt.axvline(x=float(onset), dashes=(3, 2), color=get_colormap_from_context(context='wav-hline'), lw=2)

  # layout of time plot
  if len(title): ax.set_title('Time Signal of ' + '"' + title + '"', fontsize=get_fontsize('title'))
  else: ax.set_title('Time Signal', fontsize=get_fontsize('title'))

  # labels
  ax.set_ylabel("magnitude", fontsize=get_fontsize('axis_label'))
  ax.set_xlim([0, t[-1]]), ax.grid()

  # row start and stop
  rs, re = int(n_im_rows * grid_row_usage[0]) + r_space, int(n_im_rows * 2) + r_space

  # specify grid pos
  ax = fig.add_subplot(gs[rs:re, :n_im_cols-3-2])

  # to log scale
  if log_scale: x_spec = 20 * np.log(x_spec)

  # plot image coeffs
  im = ax.imshow(x_spec, aspect='auto', interpolation='none', cmap=cmap, origin='lower')

  # frequency labels
  f_labels = ['{}k'.format(i) for i in range(9)] if not log_scale else ['0k', '1k', '4k', '8k']

  # frequency ticks positions
  f = np.arange(0, x_spec.shape[0])[::25]

  # to log scale
  if log_scale: f, _ = list(f[i] for i in [0, 1, 4, 8]), plt.yscale('symlog')
  
  # care about labels
  ax.set_xticks(np.arange(len(t_hop))[::10] - 0.5), ax.set_xticklabels(['{:.1f}'.format(ti) for ti in t_hop[::10]])
  ax.set_yticks(f), ax.set_yticklabels(f_labels, fontsize=get_fontsize('axis_tick_major'))
  ax.tick_params(axis='both', which='major', labelsize=get_fontsize('axis_tick_major')), ax.tick_params(axis='both', which='minor', labelsize=get_fontsize('axis_tick_minor'))

  # annotation
  plot_textGrid_annotation(anno_file, hop_space=fs/hop)

  # some labels
  ax.set_title('Spectrogram', fontsize=get_fontsize('title')), 
  ax.set_ylabel("frequency [Hz]", fontsize=get_fontsize('axis_label')), ax.set_xlabel("time [s]", fontsize=get_fontsize('axis_label'))

  # add colorbar
  ax = fig.add_subplot(gs[rs:re, n_im_cols-3:n_im_cols-1])
  add_colorbar(fig, im, cax=ax)

  # tight plot
  plt.subplots_adjust(left=0.1, bottom=0.00, right=0.95, top=0.93, wspace=0, hspace=0)

  # plot and close
  if plot_path is not None: plt.savefig(plot_path + name + '.png', dpi=150)

  # show plot
  if show_plot: plt.show()



if __name__ == '__main__':
  """
  main
  """
  
  import torch

  for x in [torch.randn(8, 1, 13, 20).numpy(), torch.randn(40, 1, 13, 20).numpy(), torch.randn(40, 1, 39, 20).numpy(), torch.randn(1000, 1, 13, 20).numpy()]: 
    # image plot
    plot_grid_images(x, padding=1, num_cols=8,  show_plot=False)

  plt.show()
  