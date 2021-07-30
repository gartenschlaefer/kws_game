"""
classifier class
"""

import numpy as np

# my stuff
from net_handler import NetHandler
from legacy import legacy_adjustments_net_params


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
    net_params = np.load(self.classifier_params_file, allow_pickle=True)

    # extract params
    self.nn_arch, self.train_params, self.class_dict = net_params['nn_arch'][()], net_params['train_params'][()], net_params['class_dict'][()]

    # legacy stuff
    self.data_size, self.feature_params = legacy_adjustments_net_params(net_params)

    # print info
    if self.cfg_classifier['verbose']: print("\nExtract model with architecture: [{}]\nparams: [{}]\nand class dict: [{}]".format(self.nn_arch, self.train_params, self.class_dict))
    
    # init net handler
    self.net_handler = NetHandler(nn_arch=self.nn_arch, class_dict=self.class_dict, data_size=self.data_size, feature_params=self.feature_params, use_cpu=True)

    # load model
    self.net_handler.load_models(model_files=[self.classifier_model_file])

    # set evaluation mode
    self.net_handler.set_eval_mode()

    # init to be faster
    self.classify(np.random.randn(self.net_handler.data_size[0], self.net_handler.data_size[1], self.net_handler.data_size[2]))


  def classify(self, x):
    """
    classification of a single sample
    """

    # classify
    y_hat, o, label = self.net_handler.classify_sample(x)

    # print infos
    if self.cfg_classifier['verbose']: print("\nnew sample:\nprediction: {} - {}\noutput: {}".format(y_hat, label, o.data))

    return y_hat, label
    

if __name__ == '__main__':
  """
  main of classifier
  """

  import yaml

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # config adaptions
  cfg['classifier']['verbose'] = True

  # create classifier
  classifier = Classifier(cfg_classifier=cfg['classifier'])

  # random sample
  x = np.random.randn(classifier.net_handler.data_size[0], classifier.net_handler.data_size[1], classifier.net_handler.data_size[2])

  # classify
  classifier.classify(x)