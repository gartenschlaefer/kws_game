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

    # extract params
    self.nn_arch = data['nn_arch'][()]
    self.train_params = data['train_params'][()]
    self.class_dict = data['class_dict'][()]

    # for legacy models
    try:
      self.data_size = data['data_size'][()]
    except:
      self.data_size = (1, 39, 32)
      print("old classifier model use fixed data size: {}".format(self.data_size))

    try:
      self.feature_params = data['feature_params'][()]
    except:
      self.feature_params = {'fs': 16000, 'N_s': 0.025, 'hop_s': 0.010, 'n_filter_bands': 32, 'n_ceps_coeff': 12, 'frame_size': 32, 'norm_features': False, 'use_channels': False, 'use_cepstral_features': True, 'use_delta_features': True, 'use_double_delta_features': True, 'use_energy_features': True}
      print("old classifier model use fixed feature parameters: {}".format(self.feature_params))

    # print info
    if self.cfg_classifier['verbose']:
      print("\nExtract model with architecture: [{}]\nparams: [{}]\nand class dict: [{}]".format(self.nn_arch, self.train_params, self.class_dict))
    
    # init net handler
    self.net_handler = NetHandler(nn_arch=self.nn_arch, class_dict=self.class_dict, data_size=self.data_size, use_cpu=True)

    # load model
    self.net_handler.load_models(model_files=[self.classifier_model_file])

    # set evaluation mode
    self.net_handler.set_eval_mode()

    # init to be faster
    self.classify(np.random.randn(self.net_handler.data_size[1], self.net_handler.data_size[2]))
    

  def classify(self, x):
    """
    classification of a single sample
    """

    # classify
    y_hat, o, label = self.net_handler.classify_sample(x)

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

  # config adaptions
  cfg['classifier']['verbose'] = True

  # create classifier
  classifier = Classifier(cfg_classifier=cfg['classifier'])

  # random sample
  x = np.random.randn(classifier.net_handler.data_size[1], classifier.net_handler.data_size[2])

  # classify
  classifier.classify(x)
