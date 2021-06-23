"""
Batch creation for training and testing neural networks with pytorch
"""

import numpy as np
import torch
import sys

from legacy import legacy_adjustments_feature_params


class BatchArchive():
  """
  Batch Archiv interface 
  collector of training, validation, test and my data batches
  x: data, y: label_num, z: index
  """

  #def __init__(self, batch_size=32, batch_size_eval=5, to_torch=True, shuffle=False):
  def __init__(self, batch_size_dict, to_torch=True, shuffle=False):

    # arguments
    self.batch_size_dict = batch_size_dict
    self.to_torch = to_torch
    self.shuffle = shuffle

    # batch_dict
    self.x_batch_dict = {}
    self.y_batch_dict = {}
    self.t_batch_dict = {}
    self.z_batch_dict = {}

    # class dictionary
    self.class_dict = None

    # data size
    self.data_size = None


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
    return torch.from_numpy(np.array([self.class_dict[i] for i in y])) if to_torch else np.array([self.class_dict[i] for i in y])


  def reduce_to_label(self, label):
    """
    reduce to only one label
    """

    # safety
    if label not in self.class_dict.keys():
      print("***unknown label")
      return

    # batches
    if self.y_train is not None: self.x_train, self.y_train, self.z_train = self.reduce_to_label_algorithm(label, self.x_train, self.y_train, self.z_train, self.batch_size)
    if self.y_val is not None: self.x_val, self.y_val, self.z_val = self.reduce_to_label_algorithm(label, self.x_val, self.y_val, self.z_val, self.batch_size_eval)
    if self.y_test is not None: self.x_test, self.y_test, self.z_test = self.reduce_to_label_algorithm(label, self.x_test, self.y_test, self.z_test, self.batch_size_eval)
    if self.y_my is not None: self.x_my, self.y_my, self.z_my = self.reduce_to_label_algorithm(label, self.x_my, self.y_my, self.z_my, self.batch_size_eval)

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
    if self.y_train is not None: self.x_train, self.y_train, self.z_train = self.add_noise_data_algorithm(self.x_train, self.y_train, self.z_train, noise_label, shuffle)
    if self.y_val is not None: self.x_val, self.y_val, self.z_val = self.add_noise_data_algorithm(self.x_val, self.y_val, self.z_val, noise_label, shuffle)
    if self.y_test is not None: self.x_test, self.y_test, self.z_test = self.add_noise_data_algorithm(self.x_test, self.y_test, self.z_test, noise_label, shuffle)
    if self.y_my is not None: self.x_my, self.y_my, self.z_my = self.add_noise_data_algorithm(self.x_my, self.y_my, self.z_my, noise_label, shuffle)

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
    if self.y_train is not None: self.x_train, self.y_train, self.z_train = self.one_against_all_algorithm(label, others_label, self.x_train, self.y_train, self.z_train, self.batch_size, shuffle)
    if self.y_val is not None: self.x_val, self.y_val, self.z_val = self.one_against_all_algorithm(label, others_label, self.x_val, self.y_val, self.z_val, self.batch_size_eval, shuffle)
    if self.y_test is not None: self.x_test, self.y_test, self.z_test = self.one_against_all_algorithm(label, others_label, self.x_test, self.y_test, self.z_test, self.batch_size_eval, shuffle)
    if self.y_my is not None: self.x_my, self.y_my, self.z_my = self.one_against_all_algorithm(label, others_label, self.x_my, self.y_my, self.z_my, self.batch_size_eval, shuffle)

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
  creates batches from feature files saved as .npz
  """

  def __init__(self, feature_file_dict, batch_size_dict, to_torch=True, shuffle=False):

    # parent init
    super().__init__(batch_size_dict, to_torch=to_torch, shuffle=shuffle)

    # arguments
    self.feature_file_dict = feature_file_dict

    # set names
    self.set_names = list(self.feature_file_dict.keys())

    # evaluation sets
    self.eval_set_names = self.set_names[1:]

    # load files [0]: train, etc.
    self.data_dict = {set_name: np.load(file, allow_pickle=True) for set_name, file in self.feature_file_dict.items()}

    # extract data, labels, target and index
    self.x_set_dict = {set_name: data['x'] for set_name, data in self.data_dict.items()}
    self.y_set_dict = {set_name: data['y'] for set_name, data in self.data_dict.items()}
    self.t_set_dict = {set_name: data['t'] for set_name, data in self.data_dict.items()}
    self.z_set_dict = {set_name: data['z'] for set_name, data in self.data_dict.items()}

    # extract parameters
    self.feature_params_dict = {set_name: legacy_adjustments_feature_params(data['feature_params'][()]) for set_name, data in self.data_dict.items()}
    self.cfg_dataset_dict = {set_name: data['cfg_dataset'][()] for set_name, data in self.data_dict.items()}

    # data size
    self.data_size_dict = {set_name: x.shape[1:] for set_name, x in self.x_set_dict.items()}

    # variables from first entry
    self.data_size, self.feature_params = self.data_size_dict[self.set_names[0]], self.feature_params_dict[self.set_names[0]]

    # check equivalence of variables for all sets
    if not all([d == self.data_size for d in self.data_size_dict.values()]): print("***extraction failed: data sizes are not equal"), sys.exit()
    if not all([d == self.feature_params for d in self.feature_params_dict.values()]): print("***extraction failed: feature params are not equal"), sys.exit()

    # data sizes
    self.channel_size = 1 if not self.feature_params['use_channels'] or not self.feature_params['use_mfcc_features']  else int(self.feature_params['use_cepstral_features']) + int(self.feature_params['use_delta_features']) +  int(self.feature_params['use_double_delta_features'])
    self.feature_size = (self.feature_params['n_ceps_coeff'] + int(self.feature_params['use_energy_features'])) * int(self.feature_params['use_cepstral_features']) + (self.feature_params['n_ceps_coeff'] + int(self.feature_params['use_energy_features'])) * int(self.feature_params['use_delta_features']) + (self.feature_params['n_ceps_coeff'] + int(self.feature_params['use_energy_features'])) * int(self.feature_params['use_double_delta_features']) if not self.feature_params['use_channels'] else (self.feature_params['n_ceps_coeff'] + int(self.feature_params['use_energy_features']))
    self.frame_size = self.feature_params['frame_size']
    self.raw_frame_size = int(self.feature_params['frame_size_s'] * self.feature_params['fs'])

    # check data size equivalence to parameters
    if self.feature_params['use_mfcc_features']: (print("***extraction failed: mfcc data sizes do not match to parameters"), sys.exit()) if (self.channel_size, self.feature_size, self.frame_size) != self.data_size else None, None
    else: (print("***extraction failed: raw data sizes do not match to parameters"), sys.exit()) if (self.channel_size, self.raw_frame_size) != self.data_size else None, None


    # most important class dictionary from train set
    self.class_dict = {name : i for i, name in enumerate(np.unique(self.y_set_dict['train']))}

    # init class dict only with training set
    self.class_dicts = {set_name: {name : self.class_dict[name] for i, name in enumerate(np.unique(self.y_set_dict[set_name]))} for set_name in self.set_names}

    # check if all labels from all sets are in class dict labels
    if not np.all(np.concatenate([[l in self.class_dict.keys() for l in self.class_dicts[set_name].keys()] for set_name in self.set_names])): (print("***extraction failed: labels of sets are not all in training labels"), sys.exit())
 
    # examples per class
    self.n_class_examples = {s: len(x) // len(self.class_dicts[s]) for s, x in self.x_set_dict.items()}


  def create_batches(self, selected_labels=[]):
    """
    create batches for training 
    mfcc: [n x c x m x f] -> [m x b x c x m x f]
    raw: [n x c x m] -> [m x b x c x m]
    """

    # recreate individual class dictionaries
    self.class_dicts = {set_name: {name : self.class_dict[name] for i, name in enumerate(np.unique(self.y_set_dict[set_name]))} for set_name in self.set_names}

    # go through each set
    for set_name in self.set_names:

      # copy files
      x, y, z, t = self.x_set_dict[set_name].copy(), self.y_set_dict[set_name].copy(), self.z_set_dict[set_name].copy(), self.t_set_dict[set_name].copy()

      # batch size shortcut
      batch_size = self.batch_size_dict[set_name]

      # randomize examples
      if self.shuffle:

        # random permutation
        indices = np.random.permutation(x.shape[0])

        # randomize
        x, y, z, t = np.take(x, indices, axis=0), np.take(y, indices, axis=0), np.take(z, indices, axis=0), np.take(t, indices, axis=0) if not self.feature_params['use_mfcc_features'] else None

      # reduce to labels
      if len(selected_labels) and all([l in self.class_dicts[set_name].keys() for l in selected_labels]):

        # get indices for selected labels
        label_vectors = [y == label for label in selected_labels]
        indices = np.concatenate([[i for i, lv in enumerate(label_vector) if lv] for label_vector in label_vectors])

        # take those indices
        x, y, z, t = np.take(x, indices, axis=0), np.take(y, indices, axis=0), np.take(z, indices, axis=0), np.take(t, indices, axis=0) if not self.feature_params['use_mfcc_features'] else None

        # update class dictionary
        self.class_dicts[set_name] = {name : self.class_dict[name] for i, name in enumerate(np.unique(y))}

      # number of windows
      batch_nums = x.shape[0] // batch_size

      # remaining samples
      r = int(np.remainder(x.shape[0], batch_size))

      # there are remaining samples
      if r:

        # increase batch num
        batch_nums += 1

        # indizes for remaining samples
        ss, se, random_samples = (batch_nums - 1) * batch_size, (batch_nums - 1) * batch_size + r, np.random.randint(0, high=len(y), size=batch_size-r)

        # remaining and filling examples
        r_x, r_y, r_z, f_x, f_y, f_z = x[ss:se, :], y[ss:se], z[ss:se], x[random_samples, :], y[random_samples], z[random_samples]

        # target
        r_t, f_t = (t[ss:se, :], t[random_samples, :]) if not self.feature_params['use_mfcc_features'] else (None, None)

      # init batches
      x_batches = np.empty((batch_nums, batch_size, self.channel_size, self.feature_size, self.frame_size), dtype=np.float32) if self.feature_params['use_mfcc_features'] else np.empty((batch_nums, batch_size, self.channel_size, self.raw_frame_size), dtype=np.float32)
      y_batches = np.empty((batch_nums, batch_size), dtype=np.int)
      t_batches = np.empty((batch_nums, batch_size, self.raw_frame_size), dtype=np.int) if not self.feature_params['use_mfcc_features'] else None
      z_batches = np.empty((batch_nums, batch_size), dtype=z.dtype)

      # batching
      for i in range(batch_nums - 1):
        x_batches[i, :] = x[i*batch_size:i*batch_size+batch_size, :]
        y_batches[i, :] = self.get_index_of_class(y[i*batch_size:i*batch_size+batch_size])
        z_batches[i, :] = z[i*batch_size:i*batch_size+batch_size]

        # target
        if not self.feature_params['use_mfcc_features']: t_batches[i, :] = t[i*batch_size:i*batch_size+batch_size, :]
      
      # last batch index
      i += 1

      # last batch
      x_batches[i, :] = x[i*batch_size:i*batch_size+batch_size, :] if not r else np.concatenate((r_x, f_x))
      y_batches[i, :] = self.get_index_of_class(y[i*batch_size:i*batch_size+batch_size]) if not r else self.get_index_of_class(np.concatenate((r_y, f_y)))
      z_batches[i, :] = z[i*batch_size:i*batch_size+batch_size] if not r else np.concatenate((r_z, f_z))

      # target
      if not self.feature_params['use_mfcc_features']: t_batches[i, :] = t[i*batch_size:i*batch_size+batch_size, :] if not r else np.concatenate((r_t, f_t))
      
      # to torch
      if self.to_torch: x_batches, y_batches, t_batches = torch.from_numpy(x_batches), torch.from_numpy(y_batches), torch.from_numpy(t_batches) if not self.feature_params['use_mfcc_features'] else None

      # update batch dict
      self.x_batch_dict.update({set_name: x_batches})
      self.y_batch_dict.update({set_name: y_batches})
      self.t_batch_dict.update({set_name: t_batches})
      self.z_batch_dict.update({set_name: z_batches})


    # examples per class
    self.n_class_examples = {s: np.prod(x.shape[:2]) // len(self.class_dicts[s]) for s, x in self.x_batch_dict.items()}


  def print_batch_infos(self):
    """
    prints some infos of batches
    """

    # general info
    print("\n--batch infos:")

    # go through each set
    for set_name in self.set_names:

      # print messages
      print("\nset infos: ", set_name)
      print("x: ", self.x_batch_dict[set_name].shape)
      print("y: ", self.y_batch_dict[set_name].shape)
      print("t: ", self.t_batch_dict[set_name].shape) if self.t_batch_dict[set_name] is not None else None
      print("z: ", self.z_batch_dict[set_name].shape)
      print("class dict: ", self.class_dicts[set_name])
      print("examples per class: ", self.n_class_examples[set_name])
      print("z examples: ", self.z_batch_dict[set_name][0][:20])




  # #def create_batches(self, data, batch_size=1):
  # def create_batches(self):
  #   """
  #   create batches for training 
  #   mfcc: [n x c x m x f] -> [m x b x c x m x f]
  #   raw: [n x c x m] -> [m x b x c x m]
  #   """

  #   for set_name in self.set_names

  #   # extract data
  #   x, y, z = data['x'], data['y'], data['z']

  #   # target
  #   t = data['t'] if not self.feature_params['use_mfcc_features'] else None

  #   # randomize examples
  #   if self.shuffle:

  #     # random permutation
  #     indices = np.random.permutation(x.shape[0])

  #     # randomize
  #     x, y, z = np.take(x, indices, axis=0), np.take(y, indices, axis=0), np.take(z, indices, axis=0)

  #     # target
  #     t = np.take(t, indices, axis=0) if not self.feature_params['use_mfcc_features'] else None

  #   # number of windows
  #   batch_nums = x.shape[0] // batch_size

  #   # remaining samples
  #   r = int(np.remainder(len(y), batch_size))

  #   # there are remaining samples
  #   if r:

  #     # increase batch num
  #     batch_nums += 1

  #     # indizes for remaining samples
  #     ss, se, random_samples = (batch_nums - 1) * batch_size, (batch_nums - 1) * batch_size + r, np.random.randint(0, high=len(y), size=batch_size-r)

  #     # remaining and filling examples
  #     r_x, r_y, r_z, f_x, f_y, f_z = x[ss:se, :], y[ss:se], z[ss:se], x[random_samples, :], y[random_samples], z[random_samples]

  #     # target
  #     r_t, f_t = (t[ss:se, :], t[random_samples, :]) if not self.feature_params['use_mfcc_features'] else (None, None)

  #   # init batches
  #   x_batches = np.empty((batch_nums, batch_size, self.channel_size, self.feature_size, self.frame_size), dtype=np.float32) if self.feature_params['use_mfcc_features'] else np.empty((batch_nums, batch_size, self.channel_size, self.raw_frame_size), dtype=np.float32)
  #   y_batches = np.empty((batch_nums, batch_size), dtype=np.int)
  #   t_batches = np.empty((batch_nums, batch_size, self.raw_frame_size), dtype=np.int) if not self.feature_params['use_mfcc_features'] else None
  #   z_batches = np.empty((batch_nums, batch_size), dtype=z.dtype)

  #   # batching
  #   for i in range(batch_nums - 1):
  #     x_batches[i, :] = x[i*batch_size:i*batch_size+batch_size, :]
  #     y_batches[i, :] = self.get_index_of_class(y[i*batch_size:i*batch_size+batch_size])
  #     z_batches[i, :] = z[i*batch_size:i*batch_size+batch_size]

  #     # target
  #     if not self.feature_params['use_mfcc_features']: t_batches[i, :] = t[i*batch_size:i*batch_size+batch_size, :]
    
  #   # last batch index
  #   i += 1

  #   # last batch
  #   x_batches[i, :] = x[i*batch_size:i*batch_size+batch_size, :] if not r else np.concatenate((r_x, f_x))
  #   y_batches[i, :] = self.get_index_of_class(y[i*batch_size:i*batch_size+batch_size]) if not r else self.get_index_of_class(np.concatenate((r_y, f_y)))
  #   z_batches[i, :] = z[i*batch_size:i*batch_size+batch_size] if not r else np.concatenate((r_z, f_z))

  #   # target
  #   if not self.feature_params['use_mfcc_features']: t_batches[i, :] = t[i*batch_size:i*batch_size+batch_size, :] if not r else np.concatenate((r_t, f_t))
    
  #   # to torch
  #   if self.to_torch: 

  #     x_batches, y_batches = torch.from_numpy(x_batches), torch.from_numpy(y_batches)
  #     t_batches = torch.from_numpy(t_batches) if not self.feature_params['use_mfcc_features'] else None

  #   return x_batches, y_batches, t_batches, z_batches




# --
# other functions


def plot_grid_examples(cfg, audio_set1, audio_set2):
  """
  plot examples from each label
  """

  # create batches
  batch_archive = SpeechCommandsBatchArchive(feature_file_dict={**audio_set1.feature_file_dict, **audio_set2.feature_file_dict}, batch_size_dict={'train': cfg['ml']['train_params']['batch_size'], 'test': 5, 'validation': 5, 'my': 1}, shuffle=False)

  for l in cfg['datasets']['speech_commands']['sel_labels']:

    print("l: ", l)

    # create batches
    batch_archive.create_batches(selected_labels=[l])

    # plot
    plot_grid_images(batch_archive.x_train[0, :30], context='mfcc', padding=1, num_cols=5, plot_path=cfg['datasets']['speech_commands']['plot_paths']['examples_grid'], title=l, name='grid_' + l, show_plot=False)

  # create batches for my data
  batch_archive.create_batches()
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
  batch_archive = SpeechCommandsBatchArchive(feature_file_dict={**audio_set1.feature_file_dict, **audio_set2.feature_file_dict}, batch_size_dict={'train': cfg['ml']['train_params']['batch_size'], 'test': 5, 'validation': 5, 'my': 1}, shuffle=False)

  # create batches
  batch_archive.create_batches(selected_labels=['_mixed'])

  # print info
  batch_archive.print_batch_infos()

  # all labels again
  batch_archive.create_batches()
  batch_archive.print_batch_infos()


  # plot some examples
  #plot_grid_examples(cfg, audio_set1, audio_set2)
  
  #plot_other_grid(batch_archive.x_train[0, :32], grid_size=(8, 8), show_plot=True)
  #plot_other_grid(batch_archive.x_train[-5, :32], grid_size=(8, 8), show_plot=False)
  #plot_other_grid(batch_archive.x_train[-1, :32], grid_size=(8, 8), show_plot=True)

  # x1 = batch_archive.x_train[0, 0, 0]
  # x2 = batch_archive.x_train[0, 1, 0]
  # x3 = batch_archive.x_my[0, 0, 0]

  # print("x1: ", x1.shape)
  # print("x2: ", x2.shape)
  # print("x1: ", batch_archive.z_train[0, 0])
  # print("x2: ", batch_archive.z_train[0, 1])

  # similarity measure
  #similarity_measures(x1, x2)

  #plot_mfcc_profile(x=np.ones(16000), fs=16000, N=400, hop=160, mfcc=x1)
  #plot_mfcc_only(x1, fs=16000, hop=160, plot_path=None, name=batch_archive.z_train[0, 0], show_plot=False)
  #plot_mfcc_only(x2, fs=16000, hop=160, plot_path=None, name=batch_archive.z_train[0, 1], show_plot=True)

  #plot_mfcc_equal_aspect(x2, fs=16000, hop=160, cmap=None, context='mfcc', plot_path=None, name=batch_archive.z_train[0, 1], show_plot=True)
  #plot_mfcc_equal_aspect(x3, fs=16000, hop=160, cmap=None, context='mfcc', plot_path=None, name=batch_archive.z_my[0, 0], gizmos_off=True, show_plot=True)
  #plot_mfcc_equal_aspect(x3, fs=16000, hop=160, cmap=None, context='mfcc', plot_path=None, name=batch_archive.z_my[0, 0], gizmos_off=False, show_plot=True)

  