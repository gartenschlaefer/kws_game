"""
audio datasets set creation for kws
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import librosa

from glob import glob
from shutil import copyfile

from feature_extraction import calc_mfcc39


def create_folder(paths):
	"""
	create folders in paths
	"""

	# get all folder path to create
	for p in paths:

		# if it does not exist
		if not os.path.isdir(p):

			# create path
			os.makedirs(p)


def copy_wav_files(wav_files, label, data_paths, data_percs):
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

	# get all class directories except the ones starting with _
	class_dirs = glob(dataset_path + '[!_]*/')

	# labels
	labels = [] 

	# run through all class directories
	for class_dir in class_dirs:

		# extract label
		label = class_dir.split('/')[-2]

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
		copy_wav_files(wav_files, label, data_paths, data_percs)

	return labels


def plot_mfcc(x, t, mfcc, plot_path, name='None'):
	"""
	plot mfcc extracted features from audio file
	"""
	import matplotlib.colors as colors

	# setup figure
	fig = plt.figure(figsize=(8, 8))

	# make a grid
	n_rows = 25
	n_cols = 20
	n_im_rows = 5
	gs = plt.GridSpec(n_rows, n_cols, wspace=0.4, hspace=0.3)

	# time series plot
	ax = fig.add_subplot(gs[0:n_im_rows-1, :n_cols-2])
	ax.plot(t, x)
	ax.grid()
	ax.set_title('time signal of ' + '"' + name + '"')
	ax.set_ylabel("magnitude")
	ax.set_xlim([0, t[-1]])

	# select mfcc coeffs in arrays
	sel_coefs = [np.arange(0, 12), np.arange(12, 24), np.arange(24, 36), np.arange(36, 39)]
	titles = ['12 MFCCs', 'deltas', 'double deltas', 'energies']

	# mfcc plots
	for i, c in enumerate(sel_coefs):

		# row start and stop
		rs = (i+1) * n_im_rows + 2
		re = (i+2) * n_im_rows

		# specify grid pos
		ax = fig.add_subplot(gs[rs:re, :n_cols-2])

		#im = ax.imshow(mfcc[c], aspect='auto', extent = [0, mfcc[c].shape[1], c[-1], c[0]])
		im = ax.imshow(mfcc[c], aspect='auto', extent = [0, t[-1], c[-1], c[0]])

		# color limited
		# if titles[i] != 'energies':
		# 	im = ax.imshow(mfcc[c], aspect='auto', extent = [0, t[-1], c[-1], c[0]], vmin=-100, vmax=np.max(mfcc[c]))
		#
		# else:
		# 	im = ax.imshow(mfcc[c], aspect='auto', extent = [0, t[-1], c[-1], c[0]])

		# some labels
		ax.set_title(titles[i])
		ax.set_ylabel("cepstrum coeff")
		if i == len(sel_coefs) - 1:
			ax.set_xlabel("time [s]")
		ax.set_xlim(left=0)

		# add colorbar
		ax = fig.add_subplot(gs[rs:re, n_cols-1])
		fig.colorbar(im, cax=ax)

	plt.savefig(plot_path + 'mfcc-' + name + '.png', dpi=150)


def main():
	"""
	main function
	"""

	# path to whole dataset
	dataset_path = './ignore/speech_commands_v0.01/'

	# path to training, test and eval set
	data_paths = ['./ignore/train/', './ignore/test/', './ignore/eval/']

	# wav folder
	wav_folder = 'wav/'

	# plot path
	plot_path = './ignore/plots/features/'

	# percent of data splitting [train, test], leftover is eval
	data_percs = np.array([0.6, 0.2, 0.2])

	# num examples per class
	n_examples = 10

	# create folder
	create_folder([p + wav_folder for p in data_paths] + [plot_path])

	# status message
	print("--create datasets with [{}] examples at paths:\n{}\nwith splits: {}".format(n_examples, data_paths, data_percs))

	# copy wav files to path
	labels = create_datasets(n_examples, dataset_path, [p + wav_folder for p in data_paths], data_percs)

	# list labels
	print("labels: ", labels)


	# --
	# extract mfcc features

	# sampling rate
	fs = 16000

	# mfcc analytic window
	N = int(0.025 * fs)
	#N = 512
	
	# shift of analytic window
	hop = int(0.010 * fs)

	# amount of filter bands
	n_filter_bands = 32

	# amount of first n-th cepstral coeffs
	n_ceps_coeff = 12

	# print mfcc info
	print("fs={}, mfcc: N={} is t={}, hop={} is t={}, n_f-bands={}, n_ceps_coeff={}".format(fs, N, N/fs, hop, hop/fs, n_filter_bands, n_ceps_coeff))

	# get all wav files
	#train_wavs = glob(data_paths[0] + wav_folder + '*.wav')

	# debug find specific wavs
	#train_wavs = glob(data_paths[0] + wav_folder + '*up[0-9]*.wav')
	#train_wavs = glob(data_paths[0] + wav_folder + '*up[012].wav')
	#train_wavs = glob(data_paths[0] + wav_folder + '*down[012].wav')
	train_wavs = glob(data_paths[0] + wav_folder + '*up[0].wav')

	# run through all wavs for processing
	for file_index, wav in enumerate(train_wavs):
		
		# extract label from filename
		label = re.sub(r'([0-9]+\.wav)', '', re.findall(r'[\w+ 0-9]+\.wav', wav)[0])

		# read audio from file
		x, fs = librosa.load(wav, sr=fs)

		# time vector
		t = np.arange(0, len(x)/fs, 1/fs)

		# print some info
		print("wav: [{}] with label: [{}], samples=[{}], time=[{}]s".format(wav, label, len(x), np.max(t)))

		# calculate feature vectors
		mfcc = calc_mfcc39(x, fs, N=N, hop=hop, n_filter_bands=n_filter_bands, n_ceps_coeff=n_ceps_coeff)

		# plot mfcc features
		plot_mfcc(x, t, mfcc, plot_path, name=label + str(file_index))

		# TODO: save mfcc features as .npy data for compactness store them in one data matrix with labels 


	# show plots
	plt.show()


if __name__ == '__main__':
	"""
	main caller
	"""

	# call main function
	main()


