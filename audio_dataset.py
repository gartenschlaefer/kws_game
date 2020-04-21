"""
audio datasets set creation for kws
"""

import os
import numpy as np

from glob import glob
from shutil import copyfile


def create_folder(paths, sub_folder=None):
	"""
	create train, test, eval folders
	"""

	# get all folder path to create
	for p in paths:

		# if it does not exist
		if not os.path.isdir(p):

			# create path
			os.makedirs(p)

			# create a subfolder
			if sub_folder is not None:
				os.makedirs(p + sub_folder)


def copy_wav_files(wav_files, label, data_percs, data_paths):
	"""
	copy wav files to paths with percentages
	"""

	# calculate split numbers
	n_split = np.cumsum(np.array(len(wav_files) * data_percs).astype(int))

	# actual path
	p = 0

	# run through each path
	for i, wav in enumerate(wav_files):

		# split in new path
		if i >= n_split[p]:
			p += 1

		# stop if out of range (happens at rounding errors)
		if p >= len(data_paths):
			continue

		# copy files to folder
		copyfile(wav, data_paths[p] + label + str(i) + '.wav')


def	create_datasets(n_examples, dataset_path, data_paths, data_percs):
	"""
	copy wav - files from dataset_path to data_path with spliting
	"""

	# get all class directories
	class_dirs = glob(dataset_path + '[!_]*/')

	# labels
	labels = [] 

	# run through all class directories
	for class_dir in class_dirs:

		# extract label
		label = class_dir.split('/')[-2]
		print("label: ", label)

		# append to label list
		labels.append(label)

		# get all .wav files
		wavs = glob(class_dir + '*.wav')

		# create list of wav files in class
		wav_files = []

		# get all wav files
		for i, wav in enumerate(wavs):

			# end .wav search
			if i >= n_examples:
				break

			# append wav file for copying
			wav_files.append(wav)

		# copy wav files
		copy_wav_files(wav_files, label, data_percs, [p + wav_folder for p in data_paths])

	return labels


if __name__ == '__main__':
	"""
	Main of audio dataset creator
	"""

	# path to whole dataset
	dataset_path = './ignore/speech_commands_v0.01/'

	# path to training, test and eval set
	data_paths = ['./ignore/train/', './ignore/test/', './ignore/eval/']

	# wav folder
	wav_folder = 'wav/'

	# percent of data splitting [train, test], leftover is eval
	data_percs = np.array([0.6, 0.2, 0.2])

	# num examples per class
	n_examples = 10

	# create folder
	create_folder(data_paths, sub_folder=wav_folder)

	# status message
	print("--create datasets with [{}] examples at paths:\n{}\nwith splits: {}".format(n_examples, data_paths, data_percs))

	# copy wav files to path
	labels = create_datasets(n_examples, dataset_path, data_paths, data_percs)

	print("labels: ", labels)


		
