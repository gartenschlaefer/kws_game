# --
# laboratory for unused functions, they may not even work

import yaml
import matplotlib.pyplot as plt
import numpy as np
import time

import sys
sys.path.append("../")

from audio_dataset import AudioDataset


def params():
  """
  normalize params and add noise used in ml
  """

  # normalize if requested
  if cfg['ml']['adv_params']['norm_label_weights']:
    for encoder_model, decoder_model in zip(encoder_models, decoder_models):
      with torch.no_grad():
        encoder_model.conv_layers[0].weight.div_(torch.norm(encoder_model.conv_layers[0].weight, keepdim=True))
        encoder_model.conv_layers[1].weight.div_(torch.norm(encoder_model.conv_layers[1].weight, keepdim=True))
        decoder_model.deconv_layers[0].weight.div_(torch.norm(decoder_model.deconv_layers[0].weight, keepdim=True))
        decoder_model.deconv_layers[1].weight.div_(torch.norm(decoder_model.deconv_layers[1].weight, keepdim=True))

  # add noise to each weight
  with torch.no_grad():
    for param in encoder_model.parameters():
      param.add_(torch.randn(param.shape) * 0.01)
  torch.nn.init.xavier_uniform_(collected_encoder_model.conv_layers[1].weight, gain=torch.nn.init.calculate_gain('relu'))


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

  # print
  print("o1: ", o1), print("o2: ", o2), print("o3: ", o3), print("o4: ", o4), print("o5: ", o4)


def similarity_test():
  """
  sim test
  """

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # audio sets
  audio_set1 = AudioDataset(cfg['datasets']['speech_commands'], cfg['feature_params'])
  audio_set2 = AudioDataset(cfg['datasets']['my_recordings'], cfg['feature_params'])

  # create batches
  batch_archive = SpeechCommandsBatchArchive(feature_file_dict={**audio_set1.feature_file_dict, **audio_set2.feature_file_dict}, batch_size_dict={'train': 32, 'test': 5, 'validation': 5, 'my': 1}, shuffle=False)

  # create batches of selected label
  batch_archive.create_batches(selected_labels=['_mixed'])

  # print info
  batch_archive.print_batch_infos()

  # all labels again
  batch_archive.create_batches()
  batch_archive.print_batch_infos()

  x1 = batch_archive.x_train[0, 0, 0]
  x2 = batch_archive.x_train[0, 1, 0]
  x3 = batch_archive.x_my[0, 0, 0]

  print("x1: ", x1.shape)
  print("x2: ", x2.shape)
  print("x1: ", batch_archive.z_train[0, 0])
  print("x2: ", batch_archive.z_train[0, 1])

  # similarity measure
  similarity_measures(x1, x2)


def string_compare(x):
  """
  string compare
  """
  return 1 if x == 'left' else 0


def dictionary_compare(x):
  """
  dictionary compare
  """
  return 1 if x == global_dict['left'] else 0


def num_compare(x):
  """
  number compare
  """
  return 1 if x == 1 else 0



def time_measure_callable(x, callback_f, n_samples=100):
  """
  time measurement with callable
  """

  # n measurements
  delta_time_list = []

  for i in range(n_samples):

    # measure extraction time - start
    start_time = time.time()

    # callable function
    callback_f(x)

    # result of measured time difference
    delta_time_list.append(time.time() - start_time)

  # times
  print("f: [{}] mean time: [{:.4e}]".format(callback_f.__name__, np.mean(delta_time_list)))


if __name__ == '__main__':
  """
  lab main
  """

  # similarity test
  #similarity_test()

  # global dict
  global_dict = {'left': 1}

  # time compare
  time_measure_callable('left', string_compare, n_samples=1000)
  time_measure_callable(global_dict['left'], dictionary_compare, n_samples=1000)
  time_measure_callable(1, num_compare, n_samples=1000)

