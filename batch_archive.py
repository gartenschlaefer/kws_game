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

  def __init__(self, batch_size=4, batch_size_eval=4):

    # params
    self.batch_size = batch_size
    self.batch_size_eval = batch_size_eval

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


  def create_class_dictionary(self, y):
    """
    create class dict
    """

    # actual classes
    self.classes = np.unique(y)

    # create class dict
    self.class_dict = {name : i for i, name in enumerate(self.classes)}

    # number of classes
    self.n_classes = len(self.classes)


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
      self.x_train, self.y_train = self.reduce_to_label_algorithm(label, self.x_train, self.y_train)

    # validation batches
    if self.y_val is not None:
      self.x_val, self.y_val = self.reduce_to_label_algorithm(label, self.x_val, self.y_val)

    # test batches
    if self.y_test is not None:
      self.x_test, self.y_test = self.reduce_to_label_algorithm(label, self.x_test, self.y_test)


  def reduce_to_label_algorithm(self, label, x, y):
    """
    reduce algorithm
    """

    # get label vector and feature shape
    label_vector = y == self.class_dict[label]
    f_shape = x.shape[2:]

    # get labels
    x = x[label_vector]
    y = y[label_vector]

    # reshape
    x = x[:len(x)-len(x)%self.batch_size].reshape((-1, self.batch_size) + f_shape)
    y = y[:len(y)-len(y)%self.batch_size].reshape((-1, self.batch_size))

    return x, y


  def extract(self):
    """
    extract data interface
    """
    pass



class SpeechCommandsBatchArchive(BatchArchive):
  """
  creates batches from feature files saved as .npz [train, test, eval]
  """

  def __init__(self, feature_files, batch_size=4, batch_size_eval=4, to_torch=True):

    # parent init
    super().__init__(batch_size, batch_size_eval)

    # params
    self.feature_files = feature_files
    self.to_torch = to_torch

    # load files [0]: train, etc.
    self.data = [np.load(file, allow_pickle=True) for file in self.feature_files]

    # feature params
    self.feature_params = self.data[0]['params']

    # get classes
    self.create_class_dictionary(self.data[0]['y'])

    # examples per class
    self.n_examples_class = (len(self.data[0]['x']) + len(self.data[1]['x']) + len(self.data[2]['x'])) // self.n_classes

    # do extraction
    self.extract()

    print("data: x_train: ", self.x_train.shape)


  def extract(self):
    """
    extract data samples from files
    """

    # print some infos about data
    print("\n--extract batches from data:\ntrain: {}\nval: {}\ntest: {}\n".format(self.data[0]['x'].shape, self.data[1]['x'].shape, self.data[2]['x'].shape))

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
      x_batches = torch.empty((batch_nums, batch_size, self.feature_params[()]['feature_size'], self.feature_params[()]['frame_size']))
      y_batches = torch.empty((batch_nums, batch_size), dtype=torch.long)

    else:
      x_batches = np.empty((batch_nums, batch_size, self.feature_params[()]['feature_size'], self.feature_params[()]['frame_size']), dtype=x.dtype)
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
  batch_archive = SpeechCommandsBatchArchive(audio_set1.feature_files + audio_set2.feature_files, batch_size=32, batch_size_eval=4)

  print("x_train: ", batch_archive.x_train.shape)
  print("y_train: ", batch_archive.y_train.shape)
  print("z_train: ", batch_archive.z_train.shape)

  print("x_val: ", batch_archive.x_val.shape)
  print("y_val: ", batch_archive.y_val.shape)
  print("z_val: ", batch_archive.z_val.shape)

  print("x_test: ", batch_archive.x_test.shape)
  print("y_test: ", batch_archive.y_test.shape)
  print("z_test: ", batch_archive.z_test.shape)

  print("x_my: ", batch_archive.x_my.shape)
  print("y_my: ", batch_archive.y_my.shape)
  print("z_my: ", batch_archive.z_my.shape)

  plot_mfcc_only(batch_archive.x_train[0, 0, 0], fs=16000, hop=160, plot_path=None, name=batch_archive.z_train[0, 0])

  batch_archive.reduce_to_label("up")
  print("\nreduced:")

  print("x_train: ", batch_archive.x_train.shape)
  print("y_train: ", batch_archive.y_train.shape)

  print("x_val: ", batch_archive.x_val.shape)
  print("y_val: ", batch_archive.y_val.shape)

  print("x_test: ", batch_archive.x_test.shape)
  print("y_test: ", batch_archive.y_test.shape)

  print("x_my: ", batch_archive.x_my.shape)
  print("y_my: ", batch_archive.y_my.shape)

  plot_mfcc_only(batch_archive.x_train[0, 0, 0], fs=16000, hop=160, plot_path=None, name=batch_archive.z_train[0, 0], show_plot=True)