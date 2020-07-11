"""
audio datasets set creation for kws
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import librosa

from glob import glob
from shutil import copyfile

# my stuff
from feature_extraction import calc_mfcc39, calc_onsets
from common import create_folder
from plots import plot_mfcc_profile

from torch.utils.data import Dataset, DataLoader

# TODO: Pytorch stuff
# class SpeechCommandDataset():
# 	"""
# 	Speech Command preparance for running with torch
# 	"""

# 	def __init__(self, )


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


def audio_pre_processing(wav, fs, min_samples):
	"""
	Audio pre-processing stage
	"""

	# read audio from file
	x_raw, fs = librosa.load(wav, sr=fs)

	# check sample lengths
	if len(x_raw) < min_samples:

		# print warning
		print("lengths is less than 1s, append with zeros for:")

		# append with zeros
		x_raw = np.append(x_raw, np.zeros(min_samples - len(x_raw)))

	# determine abs min value except from zero, for dithering
	min_val = np.min(np.abs(x_raw[np.abs(x_raw)>0]))

	# add some dither
	x_raw += np.random.normal(0, 0.5, len(x_raw)) * min_val

	# normalize input signal with infinity norm
	x = librosa.util.normalize(x_raw)

	return x, fs


def extract_mfcc_data(wavs, fs, N, hop, n_filter_bands, n_ceps_coeff, plot_path, ext=None, min_samples=16000, plot=True):
	"""
	extract mfcc data from wav-files
	"""

	# init mfcc_data: [n x m x l] n samples, m features, l frames
	mfcc_data = np.empty(shape=(0, 39, 98), dtype=np.float64)

	# init label list and index data
	label_data, index_data = [], []

	# run through all wavs for processing
	for wav in wavs:
		
		# extract filename
		file_name = re.findall(r'[\w+ 0-9]+\.wav', wav)[0]

		# extract file index from filename
		file_index = re.sub(r'[a-z A-Z]|(\.wav)', '', file_name)

		# extract label from filename
		label = re.sub(r'([0-9]+\.wav)', '', file_name)

		# append label and index data
		label_data.append(label)
		index_data.append(label + file_index)

		# load and pre-process audio
		x, fs = audio_pre_processing(wav, fs, min_samples)

		# print some info
		print("wav: [{}] with label: [{}], samples=[{}], time=[{}]s".format(wav, label, len(x), len(x)/fs))

		# calculate feature vectors [m x l]
		mfcc = calc_mfcc39(x, fs, N=N, hop=hop, n_filter_bands=n_filter_bands, n_ceps_coeff=n_ceps_coeff)

		# calc onsets
		onsets, onset_times = calc_onsets(x, fs, N=N, hop=hop, adapt_frames=5, adapt_alpha=0.1, adapt_beta=1)

		# plot mfcc features
		if plot:
			plot_mfcc_profile(x, fs, mfcc, plot_path, onset_times, name=label + str(file_index) + '_' + ext)

		# add to mfcc_data
		mfcc_data = np.vstack((mfcc_data, mfcc[np.newaxis, :, :]))

	return mfcc_data, label_data, index_data


if __name__ == '__main__':
	"""
	main function of audio dataset
	"""

	# path to whole dataset
	dataset_path = './ignore/speech_commands_v0.01/'

	# path to training, test and eval set
	data_paths = ['./ignore/train/', './ignore/test/', './ignore/eval/']

	# plot path
	plot_path = './ignore/plots/features/'

	# percent of data splitting [train, test], leftover is eval
	data_percs = np.array([0.8, 0.1, 0.1])

	# num examples per class
	n_examples = 10
	#n_examples = 100
	#n_examples = 500

	# wav folder
	wav_folder = 'wav_n-{}/'.format(n_examples)

	# create folder
	create_folder([p + wav_folder for p in data_paths] + [plot_path])

	# status message
	print("\n--create datasets\n[{}] examples per class saved at paths: {} with splits: {}\n".format(n_examples, data_paths, data_percs))

	# copy wav files to path
	labels = create_datasets(n_examples, dataset_path, [p + wav_folder for p in data_paths], data_percs)

	# select labels from
	# ['eight', 'sheila', 'nine', 'yes', 'one', 'no', 'left', 'tree', 'bed', 'bird', 'go', 'wow', 'seven', 'marvin', 'dog', 'three', 'two', 'house', 'down', 'six', 'five', 'off', 'right', 'cat', 'zero', 'four', 'stop', 'up', 'on', 'happy']
	sel_labels = ['left', 'right', 'up', 'down', 'go']

	# list labels
	print("labels: ", labels)
	print("\nselected labels: ", sel_labels)


	# --
	# extract mfcc features

	# sampling rate
	fs = 16000

	# mfcc window and hop size
	N, hop = int(0.025 * fs), int(0.010 * fs)

	# amount of filter bands and cepstral coeffs
	n_filter_bands, n_ceps_coeff = 32, 12

	# add params
	audio_params = {'n_examples':n_examples, 'data_percs':data_percs, 'fs':fs, 'N':N, 'hop':hop, 'n_filter_bands':n_filter_bands, 'n_ceps_coeff':n_ceps_coeff}

	# mfcc info
	mfcc_info = "n_examples={} with data split {}, fs={}, mfcc: N={} is t={}, hop={} is t={}, n_f-bands={}, n_ceps_coeff={}".format(n_examples, data_percs, fs, N, N/fs, hop, hop/fs, n_filter_bands, n_ceps_coeff)
	
	# some prints
	print(mfcc_info)
	print("params: ", audio_params)

	# for all data_paths
	for dpi, data_path in enumerate(data_paths):

		print("\ndata_path: ", data_path)

		# file extension e.g. train, test, etc.
		ext = re.sub(r'(\./\w+/)|/', '', data_path)

		# init wavs
		wavs = []

		# get all wavs
		for l in sel_labels:

			# wav re
			wav_name_re = '*' + l + '[0-9]*.wav'

			# get wavs
			wavs += glob(data_path + wav_folder + wav_name_re)[:int(data_percs[dpi] * n_examples)]

		# TODO: Only use meaning full vectors not noise

		# extract data
		mfcc_data, label_data, index_data = extract_mfcc_data(wavs, fs, N, hop, n_filter_bands, n_ceps_coeff, plot_path, ext, plot=True)

		# set file name
		file_name = '{}mfcc_data_{}_n-{}_c-{}_hop-{}.npz'.format(data_path, ext, n_examples, len(sel_labels), hop)

		# save mfcc data file
		np.savez(file_name, x=mfcc_data, y=label_data, index=index_data, info=mfcc_info, params=audio_params)

		# print
		print("--save data to: ", file_name)



	# show plots
	#plt.show()


