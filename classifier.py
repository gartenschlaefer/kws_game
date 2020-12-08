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

  def __init__(self, model_path, model_file_name='model.pth', params_file_name='params.npz', verbose=False):

    # vars
    self.model_path = model_path
    self.model_file_name = model_file_name
    self.params_file_name = params_file_name
    self.verbose = verbose

    # files
    self.model_file = self.model_path + self.model_file_name
    self.params_file = self.model_path + self.params_file_name

    # data loading
    data = np.load(self.params_file, allow_pickle=True)

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
    self.cnn_handler = CnnHandler(nn_arch=self.nn_arch, n_classes=len(self.class_dict), model_file_name=self.nn_arch, use_cpu=True)

    # load model
    self.cnn_handler.load_model(model_file=self.model_file)

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

  # create classifier
  classifier = Classifier(model_path='./models/conv-fstride/v3_c-5_n-2000/bs-32_it-1000_lr-1e-05/', model_file_name='model.pth', params_file_name='params.npz', verbose=True)

  # random sample
  x = np.random.randn(39, 32)

  # classify
  classifier.classify_sample(x)
