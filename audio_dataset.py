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
	plt.close()


def audio_pre_processing(wav, fs):
	"""
	Audio pre-processing stage
	"""

	# read audio from file
	x_raw, fs = librosa.load(wav, sr=fs)

	# determine abs min value except from zero, for dithering
	min_val = np.min(np.abs(x_raw[np.abs(x_raw)>0]))

	# add some dither
	x_raw += np.random.normal(0, 0.5, len(x_raw)) * min_val

	# normalize input signal with infinity norm
	x = librosa.util.normalize(x_raw)

	return x, fs


def extract_mfcc_data(wavs, fs, N, hop, n_filter_bands, n_ceps_coeff, plot_path, ext=None, min_samples=16000):
	"""
	extract mfcc data from wav-files
	"""

	# init mfcc_data: [n x m x t] n samples, m features, t frames
	mfcc_data = np.empty(shape=(0, 39, 98), dtype=np.float64)

	# init label list
	label_data = []

	# run through all wavs for processing
	for wav in wavs:
		
		# extract filename
		file_name = re.findall(r'[\w+ 0-9]+\.wav', wav)[0]

		# extract file index from filename
		file_index = re.sub(r'[a-z A-Z]|(\.wav)', '', file_name)

		# extract label from filename
		label = re.sub(r'([0-9]+\.wav)', '', file_name)

		# append label
		label_data.append(label)

		# load and pre-process audio
		x, fs = audio_pre_processing(wav, fs)

		# check sample lengths
		if len(x) < min_samples:

			print("file: {} lengths is less than 1s".format(file_name))
			continue

		# time vector
		t = np.arange(0, len(x)/fs, 1/fs)

		# print some info
		print("wav: [{}] with label: [{}], samples=[{}], time=[{}]s".format(wav, label, len(x), np.max(t)))

		# calculate feature vectors
		mfcc = calc_mfcc39(x, fs, N=N, hop=hop, n_filter_bands=n_filter_bands, n_ceps_coeff=n_ceps_coeff)

		# plot mfcc features
		plot_mfcc(x, t, mfcc, plot_path, name=label + str(file_index) + '_' + ext)

		# add to mfcc_data
		mfcc_data = np.vstack((mfcc_data, mfcc[np.newaxis, :, :]))

	return mfcc_data, label_data


if __name__ == '__main__':
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

	# mfcc data file
	mfcc_data_file = 'mfcc_data'

	# percent of data splitting [train, test], leftover is eval
	data_percs = np.array([0.6, 0.2, 0.2])

	# num examples per class
	n_max_examples = 10

	# create folder
	create_folder([p + wav_folder for p in data_paths] + [plot_path])

	# status message
	print("--create datasets\n[{}] examples per class saved at paths: {} with splits: {}\n".format(n_max_examples, data_paths, data_percs))

	# copy wav files to path
	labels = create_datasets(n_max_examples, dataset_path, [p + wav_folder for p in data_paths], data_percs)

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
	mfcc_info = "fs={}, mfcc: N={} is t={}, hop={} is t={}, n_f-bands={}, n_ceps_coeff={}".format(fs, N, N/fs, hop, hop/fs, n_filter_bands, n_ceps_coeff)
	print(mfcc_info)

	# get all wav files
	wav_name_re = '*.wav'

	# debug find specific wavs
	#wav_name_re = '*up[0-9]*.wav'
	#wav_name_re = '*sheila[0-9]*.wav'
	#wav_name_re = '*sheila[1]*.wav'
	#wav_name_re = '*seven[4]*.wav'

	# for all data_paths
	for data_path in data_paths:

		print("\ndata_path: ", data_path)

		# file extension e.g. train, test, etc.
		ext = re.sub(r'(\./\w+/)|/', '', data_path)

		# get wavs
		wavs = glob(data_path + wav_folder + wav_name_re)

		# extract data
		mfcc_data, label_data = extract_mfcc_data(wavs, fs, N, hop, n_filter_bands, n_ceps_coeff, plot_path, ext)

		# save file name
		file_name = data_path + mfcc_data_file + '_' + ext + '.npz'

		# save file
		np.savez(file_name, x=mfcc_data, y=label_data, info=mfcc_info)

		# print
		print("--save data to: ", file_name)

		# # load file
		# data = np.load(file_name)

		# print(data.files)
		
		# x = data['x']
		# y = data['y']
		# info = data['info']

		# print("x: ", x.shape)

	# show plots
	#plt.show()


