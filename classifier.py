"""
classifier class
"""

import numpy as np
import torch

# my stuff
from ml import get_nn_model


class Classifier():
  """
  Input Handler class
  """

  def __init__(self, file, verbose=True, root_dir=''):

    # vars
    self.file = file
    self.verbose = verbose

    # data loading
    data = np.load(self.file, allow_pickle=True)

    # print info
    print("\nextract model with params: {}\nand class dict: {}".format(data['param_str'], data['class_dict']))
    
    # extract data from file
    self.nn_arch, self.class_dict, self.path_to_file = data['params'][()]['nn_arch'], data['class_dict'][()], str(data['model_file_path'])

    # init model
    self.model = get_nn_model(self.nn_arch, n_classes=len(self.class_dict))

    # load model
    self.model.load_state_dict(torch.load(root_dir + self.path_to_file))

    # activate eval mode (no dropout layers)
    self.model.eval()


  def classify_sample(self, x):
    """
    classification by neural network
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

  # path to file
  model_path = './ignore/models/best_models/'

  # model name
  model_name = 'best_model_c-5.npz'

  # create classifier
  classifier = Classifier(file=model_path + model_name)

  # random sample
  x = np.random.randn(39, 32)

  # classify
  classifier.classify_sample(x)
