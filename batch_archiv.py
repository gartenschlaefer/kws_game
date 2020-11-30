"""
Creation of batches for training and testing neural networks with pytorch
"""

import numpy as np
import torch


class BatchArchiv():
  """
  creates batches
  """

  def __init__(self, mfcc_data_files, batch_size=4, batch_size_eval=4):

    # mfcc files saved as .npz [train, test, eval]
    self.mfcc_data_files = mfcc_data_files

    # batch sizes
    self.batch_size = batch_size
    self.batch_size_eval = batch_size_eval

    # training batches
    self.x_train = None
    self.y_train = None

    # validation batches
    self.x_val = None
    self.y_val = None

    # test batches
    self.x_test = None
    self.y_test = None

    # my batches
    self.x_my = None
    self.y_my = None
    self.z_my = None

    # load files [0]: train, etc.
    self.data = [np.load(file, allow_pickle=True) for file in self.mfcc_data_files]

    # feature params
    self.feature_params = self.data[0]['params']

    # get classes
    self.classes = np.unique(self.data[0]['y'])

    # create class dict
    self.class_dict = {name : i for i, name in enumerate(self.classes)}

    # number of classes
    self.n_classes = len(self.classes)

    # examples per class
    self.n_examples_class = (len(self.data[0]['x']) + len(self.data[1]['x']) + len(self.data[2]['x'])) // self.n_classes

    # do extraction
    self.extract()


  def extract(self):
    """
    extract data samples from files
    """

    # print some infos about data
    print("\n--extract batches from data:\ntrain: {}\nval: {}\ntest: {}\n".format(self.data[0]['x'].shape, self.data[1]['x'].shape, self.data[2]['x'].shape))

    # create batches
    self.x_train, self.y_train, _ = self.create_batches(self.data[0], batch_size=self.batch_size)
    self.x_val, self.y_val, _ = self.create_batches(self.data[1], batch_size=self.batch_size_eval)
    self.x_test, self.y_test, _ = self.create_batches(self.data[2], batch_size=self.batch_size_eval)

    # my data
    if len(self.mfcc_data_files) == 4:
      self.x_my, self.y_my, self.z_my = self.create_batches(self.data[3], batch_size=1)


  def create_batches(self, data, batch_size=1, window_step=1, plot_shift=False):
    """
    create batches for training N x [b x m x f]
    x: [n x m x l]
    y: [n]
    N: Amount of batches
    b: batch size
    m: feature size
    f: frame length
    """

    # extract data
    x_data, y_data, index, params = data['x'], data['y'], data['index'], data['params']

    # get shape of things
    n, m, l = x_data.shape

    # randomize examples
    indices = np.random.permutation(x_data.shape[0])
    x = np.take(x_data, indices, axis=0)
    y = np.take(y_data, indices, axis=0)
    z = np.take(index, indices, axis=0)

    # number of windows
    batch_nums = x.shape[0] // batch_size

    # remaining samples
    r = int(np.remainder(len(y), batch_size))
    if r:
      batch_nums += 1;

    # init batches
    x_batches = torch.empty((batch_nums, batch_size, self.feature_params[()]['feature_size'], self.feature_params[()]['frame_size']))
    y_batches = torch.empty((batch_nums, batch_size), dtype=torch.long)
    z_batches = np.empty((batch_nums, batch_size), dtype=z.dtype)

    # batching
    for i in range(batch_nums):

      # remainder handling
      if i == batch_nums - 1 and r:

        # remaining examples
        r_x = x[i*batch_size:i*batch_size+r, :]
        r_y = y[i*batch_size:i*batch_size+r]
        r_z = z[i*batch_size:i*batch_size+r]

        # pick random samples for filler
        random_samples = np.random.randint(0, high=len(y), size=batch_size-r)

        # filling examples
        f_x = x[random_samples, :]
        f_y = y[random_samples]
        f_z = z[random_samples]

        # concatenate remainder with random examples
        x_batches[i, :] = torch.from_numpy(np.concatenate((r_x, f_x)).astype(np.float32))
        y_batches[i, :] = self.get_index_of_class(np.concatenate((r_y, f_y)), to_torch=True)
        z_batches[i, :] = np.concatenate((r_z, f_z))

      # no remainder
      else:

        # get batches
        x_batches[i, :] = torch.from_numpy(x[i*batch_size:i*batch_size+batch_size, :].astype(np.float32))
        y_batches[i, :] = self.get_index_of_class(y[i*batch_size:i*batch_size+batch_size], to_torch=True)
        z_batches[i, :] = z[i*batch_size:i*batch_size+batch_size]

    # prepare for training x: [num_batches x batch_size x channel x 39 x 32]
    x_batches = torch.unsqueeze(x_batches, 2)

    return x_batches, y_batches, z_batches


  def get_index_of_class(self, y, to_torch=False):
    """
    return index of class
    """

    # init labels
    y_idx = torch.empty(y.shape, dtype=torch.long)

    for i, yi in enumerate(y):

      # get index
      idx = np.where(np.array(self.classes) == yi)[0]

      # transfer to torch
      if to_torch:
        y_idx[i] = torch.from_numpy(idx)

    return y_idx


def force_windowing(params):
  """
  windowing of data (fromer used in batches) - not in use anymore
  """
  from skimage.util.shape import view_as_windows

  # randomize data
  indices = np.random.permutation(x_data.shape[0])
  x_data = np.take(x_data, indices, axis=0)
  y_data = np.take(y_data, indices, axis=0)

  # x: [n x m x f]
  x = np.empty(shape=(0, 39, f), dtype=x_data.dtype)
  y = np.empty(shape=(0), dtype=y_data.dtype)

  # plot first example
  #fs, hop, z = params[()]['fs'], params[()]['hop'], np.take(index, indices, axis=0)
  #plot_mfcc_only(x[0], fs, hop, shift_path, name='{}-{}'.format(z[0], 0))

  # stack windows
  for i, (x_n, y_n) in enumerate(zip(x_data, y_data)):

    # windowed [r x m x f]
    x_win = np.squeeze(view_as_windows(x_n, (m, f), step=window_step))

    # window length
    l_win = x_win.shape[0]

    # append y
    y = np.append(y, [y_n] * l_win)

    # stack windowed [n+r x m x f]
    x = np.vstack((x, x_win))

    # for evaluation of shifting
    if i < 2 and plot_shift:

      # need params for plot
      fs, hop = params[()]['fs'], params[()]['hop']
      index = np.take(index, indices, axis=0)

      # plot example
      plot_mfcc_only(x_n, fs, hop, shift_path, name='{}-{}'.format(index[i], i))

      # plot some shifted mfcc
      for ri in range(x_win.shape[0]):

        # plot shifted mfcc
        plot_mfcc_only(x[ri], fs, hop, shift_path, name='{}-{}-{}'.format(index[i], i, ri))


def one_hot_label(y, classes, to_torch=False):
  """
  create one hot encoded vector e.g.:
  classes = ['up', 'down']
  y = 'up'
  return [1, 0]
  """

  # create one hot vector
  hot = np.array([c == y for c in classes]).astype(int)

  # transfer to torch
  if to_torch:
    hot = torch.from_numpy(hot)

  return hot


if __name__ == '__main__':
  """
  batching test
  """

  import yaml
  from path_collector import PathCollector

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # path collector for mfcc file pathes
  path_coll = PathCollector(cfg)

  # create batches
  batches = BatchArchiv(path_coll.mfcc_data_files_all, batch_size=32, batch_size_eval=4)

  print("x_train: ", batches.x_train.shape)
  print("y_train: ", batches.y_train.shape)

  print("x_val: ", batches.x_val.shape)
  print("y_val: ", batches.y_val.shape)

  print("x_test: ", batches.x_test.shape)
  print("y_test: ", batches.y_test.shape)

  print("x_my: ", batches.x_my.shape)
  print("y_my: ", batches.y_my.shape)