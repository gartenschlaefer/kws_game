"""
revisit metrics
"""

import numpy as np

import sys
sys.path.append("../")

from plots import plot_train_score
from score import TrainScore

if __name__ == '__main__':
	"""
	main
	"""

	metric_path = '../ignore/models/conv-encoder-fc3/v5_c-5_n-500_f-1x12x50_norm1_c1d0d0e0_nl1/bs-32_it-1000_lr-0p0001/'

	metrics = np.load(metric_path + 'metrics.npz', allow_pickle=True)

	print("metrics: ", metrics)

	# see whats in data
	print(metrics.files)

	# train_score = metrics['train_score']

	# print("train_score: ", train_score)

	# # plot train score
	# plot_train_score(train_score, plot_path=metric_path, name_ext='revisit')

