"""
Creation of batches for training and testing neural networks with pytorch
"""

import numpy as np
import torch


class BatchArchive():
  """
  Batch Archiv interface 
  collector of training, validation, test adn my data batches
  x: data, y: label_num, z: index
  """

  def __init__(self, batch_size=32, batch_size_eval=5, to_torch=True):

    # arguments
    self.batch_size = batch_size
    self.batch_size_eval = batch_size_eval
    self.to_torch = to_torch

    # training batches
    self.x_train = None
    self.y_train = None
    self.z_train = None

    # validation batches
    self.x_val = None
    self.y_val = None
    self.z_val = None

    # test batches
    self.x_test = None
    self.y_test = None
    self.z_test = None

    # my batches
    self.x_my = None
    self.y_my = None
    self.z_my = None

    # classes
    self.classes = None
    self.class_dict = None
    self.n_classes = None

    # data size
    self.data_size = None

    # number of examples per class [train, val, test, my]
    self.num_examples_per_class = [None, None, None, None]


  def create_class_dictionary(self, y):
    """
    create class dict with label names e.g. y = ['up', 'down']
    """

    # actual classes
    self.classes = np.unique(y)

    # create class dict
    self.class_dict = {name : i for i, name in enumerate(self.classes)}

    # number of classes
    self.n_classes = len(self.classes)


  def update_classes(self, new_labels):
    """
    update classes to new labels
    """

    # copy old dict
    old_dict = self.class_dict.copy()

    # recreate class directory
    self.create_class_dictionary(new_labels)

    # update y values
    for k in old_dict.keys(): 
      if k in self.class_dict.keys(): 
        if self.y_train is not None: self.y_train[self.y_train == old_dict[k]] = self.class_dict[k]
        if self.y_val is not None: self.y_val[self.y_val == old_dict[k]] = self.class_dict[k]
        if self.y_test is not None: self.y_test[self.y_test == old_dict[k]] = self.class_dict[k]
        if self.y_my is not None: self.y_my[self.y_my == old_dict[k]] = self.class_dict[k]


  def get_index_of_class(self, y, to_torch=False):
    """
    return index of class
    """

    # get index from class dict
    y_index = np.array([self.class_dict[i] for i in y])

    # to torch if necessary
    if to_torch:
      y_index = torch.from_numpy(y_index)

    return y_index


  def determine_num_examples(self):
    """
    determine number of examples in [train, val, test, my]
    """
    self.num_examples_per_class = [np.prod(y.shape) // self.n_classes for y in [self.y_train, self.y_val, self.y_test, self.y_my]]


  def reduce_to_label(self, label):
    """
    reduce to only one label
    """

    # safety
    if label not in self.class_dict.keys():
      print("***unknown label")
      return

    # training batches
    if self.y_train is not None:
      self.x_train, self.y_train, self.z_train = self.reduce_to_label_algorithm(label, self.x_train, self.y_train, self.z_train, self.batch_size)

    # validation batches
    if self.y_val is not None:
      self.x_val, self.y_val, self.z_val = self.reduce_to_label_algorithm(label, self.x_val, self.y_val, self.z_val, self.batch_size_eval)

    # test batches
    if self.y_test is not None:
      self.x_test, self.y_test, self.z_test = self.reduce_to_label_algorithm(label, self.x_test, self.y_test, self.z_test, self.batch_size_eval)

    # test batches
    if self.y_my is not None:
      self.x_my, self.y_my, self.z_my = self.reduce_to_label_algorithm(label, self.x_my, self.y_my, self.z_my, self.batch_size_eval)

    # recreate class directory
    self.update_classes([label])

    # recount examples
    self.determine_num_examples()


  def reduce_to_label_algorithm(self, label, x, y, z, batch_size):
    """
    reduce algorithm
    """

    # get label vector and feature shape
    label_vector = y == self.class_dict[label]
    f_shape = x.shape[2:]

    # get labels
    x = x[label_vector]
    y = y[label_vector]
    z = z[label_vector]

    # reshape
    x = x[:len(x)-len(x)%batch_size].reshape((-1, batch_size) + f_shape)
    y = y[:len(y)-len(y)%batch_size].reshape((-1, batch_size))
    z = z[:len(z)-len(z)%batch_size].reshape((-1, batch_size))

    return x, y, z


  def add_noise_data(self, noise_label='noise', shuffle=True):
    """
    add noise to all data
    """

    # add noise to class dict
    #self.create_class_dictionary(list(self.class_dict.keys()) + ['noise'])
    self.update_classes(list(self.class_dict.keys()) + [noise_label])

    # training batches
    if self.y_train is not None:
      self.x_train, self.y_train, self.z_train = self.add_noise_data_algorithm(self.x_train, self.y_train, self.z_train, noise_label, shuffle)

    # validation batches
    if self.y_val is not None:
      self.x_val, self.y_val, self.z_val = self.add_noise_data_algorithm(self.x_val, self.y_val, self.z_val, noise_label, shuffle)

    # test batches
    if self.y_test is not None:
      self.x_test, self.y_test, self.z_test = self.add_noise_data_algorithm(self.x_test, self.y_test, self.z_test, noise_label, shuffle)

    # test batches
    if self.y_my is not None:
      self.x_my, self.y_my, self.z_my = self.add_noise_data_algorithm(self.x_my, self.y_my, self.z_my, noise_label, shuffle)

    # recount examples
    self.determine_num_examples()


  def add_noise_data_algorithm(self, x, y, z, noise_label, shuffle=True):
    """
    add noisy batches
    """

    # torch
    if self.to_torch: 

      # vstack noise
      x = torch.as_tensor(torch.vstack((x, torch.rand(x.shape))), dtype=x.dtype)
      y = torch.as_tensor(torch.vstack((y, self.class_dict[noise_label] * torch.ones(y.shape))), dtype=y.dtype)

      z_add = z.copy()
      z_add[:] = noise_label
      z = np.vstack((z, z_add))

      # shuffle examples
      if shuffle:

        # indices for shuffling
        indices = torch.randperm(y.nelement())

        # shuffling
        x = x.view((-1,) + x.shape[2:])[indices].view(x.shape)
        y = y.view((-1,) + y.shape[2:])[indices].view(y.shape)
        z = z.reshape((-1,) + z.shape[2:])[indices].reshape(z.shape)

    return x, y, z


  def extract(self):
    """
    extract data interface
    """
    pass



class SpeechCommandsBatchArchive(BatchArchive):
  """
  creates batches from feature files saved as .npz [train, test, eval]
  """

  def __init__(self, feature_files, batch_size=32, batch_size_eval=5, to_torch=True):

    # parent init
    super().__init__(batch_size, batch_size_eval, to_torch=to_torch)

    # params
    self.feature_files = feature_files

    # load files [0]: train, etc.
    self.data = [np.load(file, allow_pickle=True) for file in self.feature_files]

    # feature params
    self.feature_params = self.data[0]['params']

    # feature size
    self.feature_size = (self.feature_params[()]['n_ceps_coeff'] + 1) * (1 + 2 * self.feature_params[()]['compute_deltas'])
    self.frame_size = self.feature_params[()]['frame_size']

    # get classes
    self.create_class_dictionary(self.data[0]['y'])

    # examples per class
    self.n_examples_class = (len(self.data[0]['x']) + len(self.data[1]['x']) + len(self.data[2]['x'])) // self.n_classes

    # do extraction
    self.extract()


  def extract(self):
    """
    extract data samples from files
    """

    # print some infos about data
    #print("\n--extract batches from data:\ntrain: {}\nval: {}\ntest: {}\n".format(self.data[0]['x'].shape, self.data[1]['x'].shape, self.data[2]['x'].shape))

    # set data size
    self.data_size = self.data[0]['x'].shape[1:]
    
    # check data sizes
    if not all([d['x'].shape[1:] == self.data_size for d in self.data]):
      print("***extraction failed: data sizes are not equal")
      return

    # add channel dimension
    self.data_size = (1, ) + self.data_size

    # create batches
    self.x_train, self.y_train, self.z_train = self.create_batches(self.data[0], batch_size=self.batch_size)
    self.x_val, self.y_val, self.z_val = self.create_batches(self.data[1], batch_size=self.batch_size_eval)
    self.x_test, self.y_test, self.z_test = self.create_batches(self.data[2], batch_size=self.batch_size_eval)

    # my data included
    if len(self.feature_files) == 4:
      self.x_my, self.y_my, self.z_my = self.create_batches(self.data[3], batch_size=1)

    # num examples
    self.determine_num_examples()


  def create_batches(self, data, batch_size=1, window_step=1, plot_shift=False):
    """
    create batches for training N x [b x m x f]
    x: [n x m x l]
    y: [n]
    N: Amount of batches
    b: batch size
    m: feature size
    f: frame size
    """

    # extract data
    x_data, y_data, z_data, params = data['x'], data['y'], data['index'], data['params']

    # get shape of things
    n, m, l = x_data.shape

    # randomize examples
    indices = np.random.permutation(x_data.shape[0])
    x = np.take(x_data, indices, axis=0)
    y = np.take(y_data, indices, axis=0)
    z = np.take(z_data, indices, axis=0)

    # number of windows
    batch_nums = x.shape[0] // batch_size

    # remaining samples
    r = int(np.remainder(len(y), batch_size))
    if r:
      batch_nums += 1;

    # init batches
    if self.to_torch:
      x_batches = torch.empty((batch_nums, batch_size, self.feature_size, self.frame_size))
      y_batches = torch.empty((batch_nums, batch_size), dtype=torch.long)

    else:
      x_batches = np.empty((batch_nums, batch_size, self.feature_size, self.frame_size), dtype=x.dtype)
      y_batches = np.empty((batch_nums, batch_size), dtype=y.dtype)

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
        # to torch if necessary (usually)
        if self.to_torch:
          x_batches[i, :] = torch.from_numpy(np.concatenate((r_x, f_x)).astype(np.float32))

        else:
          x_batches[i, :] = np.concatenate((r_x, f_x))

        y_batches[i, :] = self.get_index_of_class(np.concatenate((r_y, f_y)), to_torch=self.to_torch)
        z_batches[i, :] = np.concatenate((r_z, f_z))

      # no remainder
      else:

        # to torch if necessary (usually)
        if self.to_torch:
          x_batches[i, :] = torch.from_numpy(x[i*batch_size:i*batch_size+batch_size, :].astype(np.float32))

        else:
          # get batches
          x_batches[i, :] = x[i*batch_size:i*batch_size+batch_size, :]

        y_batches[i, :] = self.get_index_of_class(y[i*batch_size:i*batch_size+batch_size], to_torch=self.to_torch)
        z_batches[i, :] = z[i*batch_size:i*batch_size+batch_size]

    # prepare for training x: [num_batches x batch_size x channel x 39 x 32]
    if self.to_torch:
      x_batches = torch.unsqueeze(x_batches, 2)

    return x_batches, y_batches, z_batches



def print_batch_infos(batch_archive):
  """
  simply print some infos
  """

  print("x_train: ", batch_archive.x_train.shape), print("y_train: ", batch_archive.y_train.shape), print("z_train: ", batch_archive.z_train.shape), print("x_val: ", batch_archive.x_val.shape), print("y_val: ", batch_archive.y_val.shape), print("z_val: ", batch_archive.z_val.shape), print("x_test: ", batch_archive.x_test.shape), print("y_test: ", batch_archive.y_test.shape), print("z_test: ", batch_archive.z_test.shape), print("x_my: ", batch_archive.x_my.shape), print("y_my: ", batch_archive.y_my.shape), print("z_my: ", batch_archive.z_my.shape)
  print("num_examples_per_class: ", batch_archive.num_examples_per_class), print("class_dict: ", batch_archive.class_dict)
  print("y: ", batch_archive.y_train)
  print("y data type: ", batch_archive.y_train.dtype)


if __name__ == '__main__':
  """
  batching test
  """

  import yaml
  import matplotlib.pyplot as plt
  from plots import plot_mfcc_only
  from audio_dataset import AudioDataset

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # audio sets
  audio_set1 = AudioDataset(cfg['datasets']['speech_commands'], cfg['feature_params'])
  audio_set2 = AudioDataset(cfg['datasets']['my_recordings'], cfg['feature_params'])

  # create batches
  batch_archive = SpeechCommandsBatchArchive(audio_set1.feature_files + audio_set2.feature_files, batch_size=32, batch_size_eval=5)

  # infos
  #print("\ndata: "), print_batch_infos(batch_archive)

  # reduce to label
  r_label = "up"
  batch_archive.reduce_to_label(r_label)

  # infos
  #print("\nreduced to label: ", r_label), print_batch_infos(batch_archive)

  # add noise
  batch_archive.add_noise_data(shuffle=False)

  # infos
  print("\nnoise added: "), print_batch_infos(batch_archive)

  from plots import plot_grid_images, plot_other_grid

  # plot some examples
  plot_grid_images(batch_archive.x_train[0, :32], padding=1, num_cols=8, title='grid', show_plot=False)
  plot_other_grid(batch_archive.x_train[-1, :32], grid_size=(8, 8), show_plot=True)