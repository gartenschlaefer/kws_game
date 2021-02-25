"""
Machine Learning file for training and evaluating the model
"""

import numpy as np
import matplotlib.pyplot as plt

import yaml

# append paths
import sys
sys.path.append("../")

from sklearn import svm

# my stuff
from plots import plot_val_acc, plot_train_loss, plot_confusion_matrix
from batch_archive import SpeechCommandsBatchArchive
from audio_dataset import AudioDataset


if __name__ == '__main__':
  """
  SVM - as baseline
  """

  import os
  os.chdir("../")

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # audio sets
  audio_set1 = AudioDataset(cfg['datasets']['speech_commands'], cfg['feature_params'])
  audio_set2 = AudioDataset(cfg['datasets']['my_recordings'], cfg['feature_params'])


  # --
  # batches

  # create batch archiv
  batch_archive = SpeechCommandsBatchArchive(audio_set1.feature_files + audio_set2.feature_files, batch_size=1, to_torch=False)

  # print classes
  print("classes: ", batch_archive.classes)

  x = np.squeeze(batch_archive.x_train, axis=1).reshape(batch_archive.x_train.shape[0], -1)
  y = np.squeeze(batch_archive.y_train, axis=1)

  #print("x: ", x.shape)
  #print("y: ", y.shape)


  # --
  # training

  x = np.array([[1, 2], [5, 8], [1.5, 1.8], [8, 8], [1, 0.6], [9, 11]])
  y = [0, 1, 0, 1, 0, 1]

  print("x: ", x.shape)
  print("y: ", x.shape)

  clf = svm.SVC(kernel='linear', C=1.0)

  clf.fit(x, y)

  print("x: ", x[0])

  print(clf.predict(x[0].reshape(1, -1)))

  # w = clf.coef_[0]
  # print(w)

  # a = -w[0] / w[1]

  # xx = np.linspace(0,12)
  # yy = a * xx - clf.intercept_[0] / w[1]

  # h0 = plt.plot(xx, yy, 'k-', label="non weighted div")

  # plt.scatter(x[:, 0], x[:, 1], c=y)
  # plt.legend()
  # plt.show()
