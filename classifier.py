"""
classifier class
"""

import numpy as np

# my stuff
from net_handler import CnnHandler


class Classifier():
  """
  classifier class for classifying new samples with a trained model
  """

  def __init__(self, path_coll, verbose=False):

    # vars
    self.verbose = verbose

    # data loading
    data = np.load(path_coll.classifier_params, allow_pickle=True)

    # see whats in data
    #print(data.files)

    # nn architecture
    self.nn_arch = data['nn_arch'][()]
    self.train_params = data['train_params'][()]
    self.class_dict = data['class_dict'][()]

    # print info
    if verbose:
      print("\nExtract model with architecture: [{}]\nparams: [{}]\nand class dict: [{}]".format(self.nn_arch, self.train_params, self.class_dict))
    
    # init net handler
    self.cnn_handler = CnnHandler(nn_arch=self.nn_arch, n_classes=len(self.class_dict), use_cpu=True)

    # load model
    self.cnn_handler.load_model(path_coll=path_coll, for_what='classifier')

    # set evaluation mode
    self.cnn_handler.set_eval_mode()

    # init to be faster
    self.classify_sample(np.random.randn(39, 32))


  def classify_sample(self, x):
    """
    classification of a single sample
    """

    # classify
    y_hat, o = self.cnn_handler.classify_sample(x)

    # get label
    label = list(self.class_dict.keys())[list(self.class_dict.values()).index(int(y_hat))]

    # print infos
    if self.verbose:
      print("\nnew sample:\nprediction: {} - {}\noutput: {}".format(y_hat, label, o.data))

    return y_hat, label
    

if __name__ == '__main__':
  """
  main of classifier
  """

  import yaml
  from path_collector import PathCollector

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # init path collector
  path_coll = PathCollector(cfg)

  # create classifier
  classifier = Classifier(path_coll=path_coll, verbose=True)

  # random sample
  x = np.random.randn(39, 32)

  # classify
  classifier.classify_sample(x)
