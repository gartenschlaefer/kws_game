"""
classifier class
"""

import numpy as np

# my stuff
from net_handler import NetHandler


class Classifier():
  """
  classifier class for classifying new samples with a trained model
  """

  def __init__(self, cfg_classifier, root_path='./'):

    # arguments
    self.cfg_classifier = cfg_classifier
    self.root_path = root_path

    # classifier parameter file
    self.classifier_params_file = self.root_path + self.cfg_classifier['model_path'] + self.cfg_classifier['params_file_name']
    self.classifier_model_file = self.root_path + self.cfg_classifier['model_path'] + self.cfg_classifier['model_file_name']

    # data loading
    data = np.load(self.classifier_params_file, allow_pickle=True)

    # see whats in data
    print(data.files)

    # nn architecture
    self.nn_arch = data['nn_arch'][()]
    self.train_params = data['train_params'][()]
    self.class_dict = data['class_dict'][()]

    # print info
    if self.cfg_classifier['verbose']:
      print("\nExtract model with architecture: [{}]\nparams: [{}]\nand class dict: [{}]".format(self.nn_arch, self.train_params, self.class_dict))
    
    # init net handler
    self.net_handler = NetHandler(nn_arch=self.nn_arch, n_classes=len(self.class_dict), use_cpu=True)

    # load model
    self.net_handler.load_models(model_files=[self.classifier_model_file])

    # set evaluation mode
    self.net_handler.set_eval_mode()

    # init to be faster
    self.classify_sample(np.random.randn(39, 32))


  def classify_sample(self, x):
    """
    classification of a single sample
    """

    # classify
    y_hat, o = self.net_handler.classify_sample(x)

    # get label
    label = list(self.class_dict.keys())[list(self.class_dict.values()).index(int(y_hat))]

    # print infos
    if self.cfg_classifier['verbose']:
      print("\nnew sample:\nprediction: {} - {}\noutput: {}".format(y_hat, label, o.data))

    return y_hat, label
    

if __name__ == '__main__':
  """
  main of classifier
  """

  import yaml

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # create classifier
  classifier = Classifier(cfg_classifier=cfg['classifier'])

  # random sample
  x = np.random.randn(39, 32)

  # classify
  classifier.classify_sample(x)
