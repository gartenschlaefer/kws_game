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
    if to_torch: y_index = torch.from_numpy(y_index)

    return y_index


  def determine_num_examples(self):
    """
    determine number of examples in [train, val, test, my]
    """
    self.num_examples_per_class = [np.prod(y.shape) // self.n_classes if y is not None else 0 for y in [self.y_train, self.y_val, self.y_test, self.y_my]]


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


  def reduce_to_labels(self, labels):
    """
    reduce to labels (must be an array)
    """

    # safety
    if any([label not in self.class_dict.keys() for label in labels]):
      print("***unknown label")
      return

    # reduce batches
    if self.y_train is not None: self.x_train, self.y_train, self.z_train = self.reduce_to_labels_algorithm(labels, self.x_train, self.y_train, self.z_train, self.batch_size)
    if self.y_val is not None: self.x_val, self.y_val, self.z_val = self.reduce_to_labels_algorithm(labels, self.x_val, self.y_val, self.z_val, self.batch_size_eval)
    if self.y_test is not None: self.x_test, self.y_test, self.z_test = self.reduce_to_labels_algorithm(labels, self.x_test, self.y_test, self.z_test, self.batch_size_eval)
    if self.y_my is not None: self.x_my, self.y_my, self.z_my = self.reduce_to_labels_algorithm(labels, self.x_my, self.y_my, self.z_my, self.batch_size_eval)

    # recreate class directory
    self.update_classes(labels)

    # recount examples
    self.determine_num_examples()


  def reduce_to_labels_algorithm(self, labels, x, y, z, batch_size, shuffle=True):
    """
    reduce algorithm
    """

    # get label vector and feature shape
    label_vectors = [y == self.class_dict[label] for label in labels]

    # logical or the label vectors
    all_label_vectors = torch.zeros(size=label_vectors[0].shape)
    for label_vector in label_vectors: all_label_vectors = torch.logical_or(all_label_vectors, label_vector)
    f_shape = x.shape[2:]

    # get labels
    x = x[all_label_vectors]
    y = y[all_label_vectors]
    z = z[all_label_vectors]

    # reshape
    x = x[:len(x)-len(x)%batch_size].reshape((-1, batch_size) + f_shape)
    y = y[:len(y)-len(y)%batch_size].reshape((-1, batch_size))
    z = z[:len(z)-len(z)%batch_size].reshape((-1, batch_size))

    # shuffle examples
    if shuffle: x, y, z = self.xyz_shuffle(x, y, z)

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

    # my batches
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
      if shuffle: x, y, z = self.xyz_shuffle(x, y, z)

    return x, y, z


  def xyz_shuffle(self, x, y, z):
    """
    shuffle
    """

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


  def one_against_all(self, label, others_label='other', shuffle=True):
    """
    one against all
    """

    # safety
    if label not in self.class_dict.keys():
      print("***unknown label")
      return

    #self.update_classes(list(self.class_dict.keys()) + [others_label])
    self.class_dict.update({others_label:9999})

    # training batches
    if self.y_train is not None:
      self.x_train, self.y_train, self.z_train = self.one_against_all_algorithm(label, others_label, self.x_train, self.y_train, self.z_train, self.batch_size, shuffle)

    # validation batches
    if self.y_val is not None:
      self.x_val, self.y_val, self.z_val = self.one_against_all_algorithm(label, others_label, self.x_val, self.y_val, self.z_val, self.batch_size_eval, shuffle)

    # test batches
    if self.y_test is not None:
      self.x_test, self.y_test, self.z_test = self.one_against_all_algorithm(label, others_label, self.x_test, self.y_test, self.z_test, self.batch_size_eval, shuffle)

    # my batches
    if self.y_my is not None:
      self.x_my, self.y_my, self.z_my = self.one_against_all_algorithm(label, others_label, self.x_my, self.y_my, self.z_my, self.batch_size_eval, shuffle)

    # recreate class directory
    self.update_classes([label, others_label])

    # recount examples
    self.determine_num_examples()


  def one_against_all_algorithm(self, label, others_label, x, y, z, batch_size, shuffle=True):
    """
    one against all algorithm
    """

    # get label vector and feature shape
    label_vector = y == self.class_dict[label]
    f_shape = x.shape[2:]

    other_labels = list(self.class_dict.keys())
    other_labels.remove(label)
    other_labels.remove(others_label)

    # one and others
    x_one, y_one, z_one = x[label_vector], y[label_vector], z[label_vector]
    x_other, y_other, z_other = torch.empty((0, batch_size) + f_shape), torch.empty((0, batch_size)), np.empty((0, batch_size))

    # reshape
    x_one = x_one[:len(x_one)-len(x_one)%batch_size].reshape((-1, batch_size) + f_shape)
    y_one = y_one[:len(y_one)-len(y_one)%batch_size].reshape((-1, batch_size))
    z_one = z_one[:len(z_one)-len(z_one)%batch_size].reshape((-1, batch_size))

    # others
    for l in other_labels:

      # label vector
      label_vector = y == self.class_dict[l]

      # get labeled data
      x_l, y_l, z_l = x[label_vector], y[label_vector], z[label_vector]

      # reshape
      x_l = x_l[:len(x_l)-len(x_l)%batch_size].reshape((-1, batch_size) + f_shape)
      y_l = y_l[:len(y_l)-len(y_l)%batch_size].reshape((-1, batch_size))
      z_l = z_l[:len(z_l)-len(z_l)%batch_size].reshape((-1, batch_size))

      # label to others label
      y_l[:] = self.class_dict[others_label]

      # concatenate
      x_other = torch.vstack((x_other, x_l[:int(np.ceil(y_one.shape[0] / (self.n_classes - 1)))]))
      y_other = torch.vstack((y_other, y_l[:int(np.ceil(y_one.shape[0] / (self.n_classes - 1)))]))
      z_other = np.vstack((z_other, z_l[:int(np.ceil(y_one.shape[0] / (self.n_classes - 1)))]))

    # stack one and other
    x = torch.as_tensor(torch.vstack((x_one, x_other)), dtype=x.dtype)
    y = torch.as_tensor(torch.vstack((y_one, y_other)), dtype=y.dtype)
    z = np.vstack((z_one, z_other))

    # shuffle examples
    if shuffle: x, y, z = self.xyz_shuffle(x, y, z)

    return x, y, z 



class SpeechCommandsBatchArchive(BatchArchive):
  """
  creates batches from feature files saved as .npz [train, test, eval]
  """

  def __init__(self, feature_files, batch_size=32, batch_size_eval=5, to_torch=True, shuffle=True):

    # parent init
    super().__init__(batch_size, batch_size_eval, to_torch=to_torch)

    # params
    self.feature_files = feature_files
    self.shuffle = shuffle

    # load files [0]: train, etc.
    self.data = [np.load(file, allow_pickle=True) for file in self.feature_files]

    # feature params
    self.feature_params = self.data[0]['params'][()]

    # channel size
    self.channel_size = 1 if not self.feature_params['use_channels'] else int(self.feature_params['use_cepstral_features']) + int(self.feature_params['use_delta_features']) +  int(self.feature_params['use_double_delta_features'])

    # feature size
    self.feature_size = (self.feature_params['n_ceps_coeff'] + int(self.feature_params['use_energy_features'])) * int(self.feature_params['use_cepstral_features']) + (self.feature_params['n_ceps_coeff'] + int(self.feature_params['use_energy_features'])) * int(self.feature_params['use_delta_features']) + (self.feature_params['n_ceps_coeff'] + int(self.feature_params['use_energy_features'])) * int(self.feature_params['use_double_delta_features']) if not self.feature_params['use_channels'] else (self.feature_params['n_ceps_coeff'] + int(self.feature_params['use_energy_features']))

    # frame size
    self.frame_size = self.feature_params['frame_size']

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

    # set data size
    self.data_size = self.data[0]['x'].shape[1:]
    
    # check data sizes
    if not all([d['x'].shape[1:] == self.data_size for d in self.data]): 
      print("***extraction failed: data sizes are not equal")
      return

    # create batches
    self.x_train, self.y_train, self.z_train = self.create_batches(self.data[0], batch_size=self.batch_size)
    self.x_val, self.y_val, self.z_val = self.create_batches(self.data[1], batch_size=self.batch_size_eval)
    self.x_test, self.y_test, self.z_test = self.create_batches(self.data[2], batch_size=self.batch_size_eval)

    # my data included
    if len(self.feature_files) == 4:
      self.x_my, self.y_my, self.z_my = self.create_batches(self.data[3], batch_size=1)

    # num examples
    self.determine_num_examples()


  def create_batches(self, data, batch_size=1):
    """
    create batches for training [n x c x m x f] -> [m x b x c x m x f]
    """

    # extract data
    x, y, z, params = data['x'], data['y'], data['index'], data['params']

    # get shape of things
    n, c, m, l = x.shape

    # randomize examples
    if self.shuffle:
      indices = np.random.permutation(x.shape[0])
      x, y, z = np.take(x, indices, axis=0), np.take(y, indices, axis=0), np.take(z, indices, axis=0)

    # number of windows
    batch_nums = x.shape[0] // batch_size

    # remaining samples
    r = int(np.remainder(len(y), batch_size))

    # there are remaining samples
    if r:

      # increase batch num
      batch_nums += 1

      # indizes for remaining samples
      ss, se, random_samples = (batch_nums - 1) * batch_size, (batch_nums - 1) * batch_size + r, np.random.randint(0, high=len(y), size=batch_size-r)

      # remaining and filling examples
      r_x, r_y, r_z, f_x, f_y, f_z = x[ss:se, :], y[ss:se], z[ss:se], x[random_samples, :], y[random_samples], z[random_samples]

    # init batches
    x_batches = np.empty((batch_nums, batch_size, self.channel_size, self.feature_size, self.frame_size), dtype=np.float32)
    y_batches = np.empty((batch_nums, batch_size), dtype=np.int)
    z_batches = np.empty((batch_nums, batch_size), dtype=z.dtype)

    # batching
    for i in range(batch_nums - 1):
      x_batches[i, :] = x[i*batch_size:i*batch_size+batch_size, :]
      y_batches[i, :] = self.get_index_of_class(y[i*batch_size:i*batch_size+batch_size])
      z_batches[i, :] = z[i*batch_size:i*batch_size+batch_size]

    # last batch index
    i += 1

    # last batch
    x_batches[i, :] = x[i*batch_size:i*batch_size+batch_size, :] if not r else np.concatenate((r_x, f_x))
    y_batches[i, :] = self.get_index_of_class(y[i*batch_size:i*batch_size+batch_size]) if not r else self.get_index_of_class(np.concatenate((r_y, f_y)))
    z_batches[i, :] = z[i*batch_size:i*batch_size+batch_size] if not r else np.concatenate((r_z, f_z))

    # to torch
    if self.to_torch: x_batches, y_batches = torch.from_numpy(x_batches), torch.from_numpy(y_batches)

    return x_batches, y_batches, z_batches



def print_batch_infos(batch_archive):
  """
  simply print some infos
  """

  print("x_train: ", batch_archive.x_train.shape), print("y_train: ", batch_archive.y_train.shape), print("z_train: ", batch_archive.z_train.shape), print("x_val: ", batch_archive.x_val.shape), print("y_val: ", batch_archive.y_val.shape), print("z_val: ", batch_archive.z_val.shape), print("x_test: ", batch_archive.x_test.shape), print("y_test: ", batch_archive.y_test.shape), print("z_test: ", batch_archive.z_test.shape), print("x_my: ", batch_archive.x_my.shape), print("y_my: ", batch_archive.y_my.shape), print("z_my: ", batch_archive.z_my.shape)
  print("num_examples_per_class: ", batch_archive.num_examples_per_class), print("class_dict: ", batch_archive.class_dict)
  print("y: ", batch_archive.y_train)
  print("y data type: ", batch_archive.y_train.dtype)
  print("y: ", batch_archive.y_test)
  print("y_my: ", batch_archive.y_my)


def plot_grid_examples(cfg, audio_set1, audio_set2):
  """
  plot examples from each label
  """

  for l in cfg['datasets']['speech_commands']['sel_labels']:

    # create batches
    batch_archive = SpeechCommandsBatchArchive(audio_set1.feature_files + audio_set2.feature_files, batch_size=32, batch_size_eval=5)

    # reduce to label
    batch_archive.reduce_to_label(l)

    print("l: ", l)

    # plot
    plot_grid_images(batch_archive.x_train[0, :30], context='mfcc', padding=1, num_cols=5, plot_path=cfg['datasets']['speech_commands']['plot_paths']['examples_grid'], title=l, name='grid_' + l, show_plot=False)

  # create batches for my data
  batch_archive = SpeechCommandsBatchArchive(audio_set1.feature_files + audio_set2.feature_files, batch_size=32, batch_size_eval=5, shuffle=False)
  print("\ndata: "), print_batch_infos(batch_archive)
  
  # plot my data
  plot_grid_images(np.squeeze(batch_archive.x_my, axis=1), context='mfcc', padding=1, num_cols=5, plot_path=cfg['datasets']['my_recordings']['plot_paths']['examples_grid'], title='grid', name='grid', show_plot=False)


def similarity_measures(x1, x2):
  """
  similarities
  """

  # noise
  n1, n2 = torch.randn(x1.shape), torch.randn(x2.shape)

  # similarity
  cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-08)

  # similarity measure
  o1, o2, o3, o4, o5 = cos_sim(x1, x1), cos_sim(x1, x2), cos_sim(n1, n2), cos_sim(x1, n1), cos_sim(x2, n2)

  # cosine sim definition
  #o = x1[0] @ x2[0].T / np.max(np.linalg.norm(x1[0]) * np.linalg.norm(x2[0]))

  print("o1: ", o1), print("o2: ", o2), print("o3: ", o3), print("o4: ", o4), print("o5: ", o4)


if __name__ == '__main__':
  """
  batching test
  """

  import yaml
  import matplotlib.pyplot as plt
  from plots import plot_mfcc_only, plot_grid_images, plot_other_grid, plot_mfcc_profile, plot_mfcc_equal_aspect
  from audio_dataset import AudioDataset

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # audio sets
  audio_set1 = AudioDataset(cfg['datasets']['speech_commands'], cfg['feature_params'])
  audio_set2 = AudioDataset(cfg['datasets']['my_recordings'], cfg['feature_params'])

  # create batches
  batch_archive = SpeechCommandsBatchArchive(audio_set1.feature_files + audio_set2.feature_files, batch_size=32, batch_size_eval=5, shuffle=False)

  # reduce to labels algorithm
  batch_archive.reduce_to_labels(['left', 'right'])

  # infos
  print("\ndata: "), print_batch_infos(batch_archive)

  # reduce to label
  #r_label = "up"
  #batch_archive.reduce_to_label(r_label)

  # infos
  #print("\nreduced to label: ", r_label), print_batch_infos(batch_archive)

  # add noise
  #batch_archive.add_noise_data(shuffle=False)

  # infos
  #print("\nnoise added: "), print_batch_infos(batch_archive)

  #batch_archive.one_against_all(r_label, others_label='other', shuffle=False)
  #print("\none against all: "), print_batch_infos(batch_archive)


  # plot some examples
  #plot_grid_examples(cfg, audio_set1, audio_set2)
  

  #plot_other_grid(batch_archive.x_train[0, :32], grid_size=(8, 8), show_plot=True)
  #plot_other_grid(batch_archive.x_train[-5, :32], grid_size=(8, 8), show_plot=False)
  #plot_other_grid(batch_archive.x_train[-1, :32], grid_size=(8, 8), show_plot=True)

  x1 = batch_archive.x_train[0, 0, 0]
  x2 = batch_archive.x_train[0, 1, 0]
  x3 = batch_archive.x_my[0, 0, 0]

  print("x1: ", x1.shape)
  print("x2: ", x2.shape)
  print("x1: ", batch_archive.z_train[0, 0])
  print("x2: ", batch_archive.z_train[0, 1])

  # similarity measure
  #similarity_measures(x1, x2)


  #plot_mfcc_profile(x=np.ones(16000), fs=16000, N=400, hop=160, mfcc=x1)
  plot_mfcc_only(x1, fs=16000, hop=160, plot_path=None, name=batch_archive.z_train[0, 0], show_plot=False)
  plot_mfcc_only(x2, fs=16000, hop=160, plot_path=None, name=batch_archive.z_train[0, 1], show_plot=True)

  #plot_mfcc_equal_aspect(x2, fs=16000, hop=160, cmap=None, context='mfcc', plot_path=None, name=batch_archive.z_train[0, 1], show_plot=True)
  #plot_mfcc_equal_aspect(x3, fs=16000, hop=160, cmap=None, context='mfcc', plot_path=None, name=batch_archive.z_my[0, 0], gizmos_off=True, show_plot=True)
  #plot_mfcc_equal_aspect(x3, fs=16000, hop=160, cmap=None, context='mfcc', plot_path=None, name=batch_archive.z_my[0, 0], gizmos_off=False, show_plot=True)

  