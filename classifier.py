"""
classifier class
"""

import numpy as np
import torch

# my stuff
from ml import get_nn_model


class Classifier():
  """
  classifier class for classifying new samples with a trained model
  file: path to .pth file
  """

  def __init__(self, file, verbose=False):

    # vars
    self.file = file
    self.verbose = verbose

    # root path
    self.root_path = '/'.join(file.split('/')[:-1]) + '/'

    # data loading
    data = np.load(self.file, allow_pickle=True)

    # print info
    print("\nextract model with params: {}\nand class dict: {}".format(data['param_str'], data['class_dict']))
    
    # extract data from file
    self.nn_arch, self.class_dict, self.path_to_file = data['params'][()]['nn_arch'], data['class_dict'][()], self.root_path + str(data['model_file_path']).split('/')[-1]

    # init model
    self.model = get_nn_model(self.nn_arch, n_classes=len(self.class_dict))

    # load model
    self.model.load_state_dict(torch.load(self.path_to_file))

    # activate eval mode (no dropout layers)
    self.model.eval()

    # init to be faster
    self.classify_sample(np.random.randn(39, 32))


  def classify_sample(self, x):
    """
    classification of a single sample
    """

    # input to tensor
    x = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(x.astype(np.float32)), 0), 0)

    # no gradients for eval
    with torch.no_grad():

      # classify
      o = self.model(x)

      # prediction
      _, y_hat = torch.max(o.data, 1)

      if self.verbose:
        print("\nnew sample:\nprediction: {} - {}\noutput: {}".format(y_hat, list(self.class_dict.keys())[list(self.class_dict.values()).index(int(y_hat))], o.data))

    return int(y_hat)


if __name__ == '__main__':
  """
  main of classifier
  """

  # create classifier
  classifier = Classifier(file='./models/fstride_c-5.npz', verbose=True)

  # random sample
  x = np.random.randn(39, 32)

  # classify
  classifier.classify_sample(x)
