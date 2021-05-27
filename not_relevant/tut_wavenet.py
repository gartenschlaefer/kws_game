"""
wavenet tutorials
"""

import numpy as np
import torch

# append paths
import sys
sys.path.append("../")
from wavenet import Wavenet


def wavenet_training(wavenet, x_train, y_train, class_dict, num_epochs=10, lr=0.0001):
  """
  wavenet training
  """

  # dataseize
  data_size = x_train.shape[1:]

  # cpu
  use_cpu = True

  # vars
  n_classes = len(class_dict)
  num_print_per_epoch = 2

  # set device
  device = torch.device("cuda:0" if (torch.cuda.is_available() and not use_cpu) else "cpu")

  # criterion
  criterion = torch.nn.CrossEntropyLoss()

  # optimizer
  optimizer = torch.optim.Adam(wavenet.parameters(), lr=lr)


  print("\n--Training starts:")

  batch_loss = 0
  k_print = 1

  # epochs
  for epoch in range(num_epochs):

    # fetch data samples
    for i, (x, y) in enumerate(zip(x_train.to(device), y_train.to(device))):

      # target
      t = x



      # zero parameter gradients
      optimizer.zero_grad()

      # forward pass o: [b x c x samples]
      o = wavenet(x)

      #print("x: ", x)
      #print("x: ", x.shape)

      #print("o1: ", o)
      #print("o1: ", o.shape)

      # reshape
      o = o.view((-1, o.shape[1]))
      t = t.view(-1)

      #print("o2: ", o)
      #print("o2: ", o.shape)

      # quantize data
      t = wavenet.quantize(t, n_classes=256)

      #print("t: ", t)
      #print("t: ", t.shape)

      # loss
      loss = criterion(o, t)

      # backward
      loss.backward()

      # optimizer step - update params
      optimizer.step()

      # batch loss
      batch_loss += loss.item()

      # print loss
      if i % k_print == k_print-1:

        # print info
        print('epoch: {}, mini-batch: {}, loss: [{:.5f}]'.format(epoch + 1, i + 1, batch_loss / k_print))
        batch_loss = 0

  print('--Training finished')



if __name__ == '__main__':
  """
  main
  """

  # input [num x batch x channel x time]
  x_train = torch.randn(1, 1, 1, 16000)
  y_train = torch.randint(0, 2, (1, 1, 16000))

  class_dict = {'a':0, 'b':1}

  # norm
  x_train = x_train / torch.max(torch.abs(x_train))

  # wavenet
  wavenet = Wavenet()
  print("wavenet: ", wavenet)

  # wavenet training
  wavenet_training(wavenet, x_train, y_train, class_dict, num_epochs=10, lr=0.001)

  # count params
  layer_params = wavenet.count_params()
  print("layer: {} sum: {}".format(layer_params, sum(layer_params)))
