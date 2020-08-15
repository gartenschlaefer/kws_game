"""
Insight file for investigating the nn models
"""

import numpy as np
import matplotlib.pyplot as plt

import torch

# my stuff
from common import *
from plots import *
from conv_nets import *

from ml import get_nn_model, get_pretrained_model


if __name__ == '__main__':
  """
  Insight file - gives insight in neural nets
  """

  # plot path and model path
  plot_path, model_pre_path = './ignore/plots/insight/', './ignore/models/pre/'

  # create folder
  create_folder([plot_path])

  # pretrained model
  pre_trained_model_path = model_pre_path + 'conv-fstride_c-30.pth'

    # nn architecture
  nn_architectures = ['conv-trad', 'conv-fstride']

  # select model
  nn_arch, n_classes = nn_architectures[1], 30

  # init model
  model = get_nn_model(nn_arch='conv-fstride', n_classes=n_classes)

  # use pre-trained model
  model = get_pretrained_model(model, pre_trained_model_path)

  print("\nmodel:\n", model)

  print("weights: ", model.fc1.weight.detach())
  print("weights: ", model.fc1.weight.shape)

  print("conv weights: ", model.conv.weight.detach())
  print("conv weights: ", model.conv.weight.detach().shape)

  plt.figure(), plt.imshow(np.squeeze(model.conv.weight.detach().numpy()[0]), aspect='auto'), plt.colorbar()
  plt.figure(), plt.imshow(np.squeeze(model.conv.weight.detach().numpy()[1]), aspect='auto'), plt.colorbar()
  plt.figure(), plt.imshow(np.squeeze(model.conv.weight.detach().numpy()[2]), aspect='auto'), plt.colorbar()
  plt.figure(), plt.imshow(np.squeeze(model.conv.weight.detach().numpy()[3]), aspect='auto'), plt.colorbar()

  #plt.figure(), plt.imshow(model.fc1.weight.detach().numpy(), aspect='auto'), plt.colorbar()
  #plt.figure(), plt.imshow(model.fc2.weight.detach().numpy(), aspect='auto'), plt.colorbar()
  #plt.figure(), plt.imshow(model.fc4.weight.detach().numpy(), aspect='auto'), plt.colorbar()

  plt.show()



