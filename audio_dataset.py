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
from feature_extraction import calc_mfcc39, calc_onsets, frames_to_time
from common import create_folder
from plots import *

from torch.utils.data import Dataset, DataLoader
from skimage.util.shape import view_as_windows

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
			break

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


def extract_mfcc_data(wavs, params, frame_size=32, ext=None, min_samples=16000, plot_path=None):
	"""
	extract mfcc data from wav-files
	"""

	# extract params
	fs, N, hop, n_filter_bands, n_ceps_coeff = params['fs'], params['N'], params['hop'], params['n_filter_bands'], params['n_ceps_coeff']

	# init mfcc_data: [n x m x l] n samples, m features, l frames
	mfcc_data = np.empty(shape=(0, 39, frame_size), dtype=np.float64)
	
	# damaged file score
	z_score_list, broken_file_list = np.array([]), []

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

		# load and pre-process audio
		x, fs = audio_pre_processing(wav, fs, min_samples)

		# print some info
		print("wav: [{}] with label: [{}], samples=[{}], time=[{}]s".format(wav, label, len(x), len(x)/fs))

		# calculate feature vectors [m x l]
		mfcc = calc_mfcc39(x, fs, N=N, hop=hop, n_filter_bands=n_filter_bands, n_ceps_coeff=n_ceps_coeff)

		# calc onsets
		#onsets = calc_onsets(x, fs, N=N, hop=hop, adapt_frames=5, adapt_alpha=0.1, adapt_beta=1)
		onsets = calc_onsets(x, fs, N=N, hop=hop, adapt_frames=5, adapt_alpha=0.05, adapt_beta=0.9)

		# find best onset
		#best_onset, bon_pos = find_best_onset(onsets, frame_size=frame_size, pre_frames=1)
		mient = find_min_energy_time(mfcc, fs, hop)
		minreg, bon_pos = find_min_energy_region(mfcc, fs, hop)

		# damaged file things
		z_score, z_damaged = detect_damaged_file(mfcc)
		z_score_list = np.append(z_score_list, z_score)

		# plot mfcc features
		plot_mfcc_profile(x, fs, N, hop, mfcc, onsets, bon_pos, mient, minreg, frame_size, plot_path, name=label + str(file_index) + '_' + ext)

		# handled damaged files
		if z_damaged:
			print("--*file probably broken!")
			broken_file_list.append(file_name)
			continue

		# add to mfcc_data
		mfcc_data = np.vstack((mfcc_data, mfcc[np.newaxis, :, bon_pos:bon_pos+frame_size]))
		label_data.append(label)
		index_data.append(label + file_index)

	# broken file info
	plot_damaged_file_score(z_score_list, plot_path=plot_path, name='z_score_n-{}'.format(params['n_examples']))
	print("\n --broken file list: \n{}\nwith length: {}".format(broken_file_list, len(broken_file_list)))

	return mfcc_data, label_data, index_data


def detect_damaged_file(mfcc, z_lim=60):
	"""
	detect if file is damaged
	"""

	# calculate damaged score of energy deltas
	z_est = np.sum(mfcc[37:39, :])

	# return score and damaged indicator
	return z_est, z_est > z_lim


def find_min_energy_region(mfcc, fs, hop, frame_size=32):
	"""
	find frame with least amount of energy
	"""

	# windowed [r x m x f]
	x_win = np.squeeze(view_as_windows(mfcc[36, :], frame_size, step=1))

	# best onset position
	bon_pos = np.argmin(np.sum(x_win, axis=1))

	return frames_to_time(bon_pos, fs, hop), bon_pos


def find_min_energy_time(mfcc, fs, hop):
	"""
	find min  energy time position
	"""

	return frames_to_time(np.argmin(mfcc[36, :]), fs, hop)


def find_best_onset(onsets, frame_size=32, pre_frames=1):
	"""
	find the best onset with highest propability of spoken word
	"""

	# init
	best_onset, bon_pos = np.zeros(onsets.shape), 0

	# determine onset positions
	onset_pos = np.squeeze(np.argwhere(onsets))

	# single onset handling
	if int(np.sum(onsets)) == 1:

		#return onsets, int(np.where(onsets == 1)[0][0])
		best_onset = onsets
		bon_pos = onset_pos

	# multiple onsets handling
	else: 

		# windowing
		o_win = view_as_windows(np.pad(onsets, (0, frame_size-1)), window_shape=(frame_size), step=1)[onset_pos, :]

		# get index of best onset
		x_max = np.argmax(np.sum(o_win, axis=1))

		# set single best onset
		bon_pos = onset_pos[x_max]
		best_onset[bon_pos] = 1

	# pre frames before real onset
	if bon_pos - pre_frames > 0:
		best_onset = np.roll(best_onset, -pre_frames)

	# best onset on right egde, do roll
	if bon_pos - pre_frames >= (onsets.shape[0] - frame_size):
		r = frame_size - (onsets.shape[0] - (bon_pos - pre_frames)) + 1
		best_onset = np.roll(best_onset, -r)

	#print("best_onset: ", best_onset)
	#print("pos: ", int(np.where(best_onset == 1)[0][0]))

	return best_onset, int(np.where(best_onset == 1)[0][0])


def label_stats(y):
	"""
	labels
	"""

	labels = np.unique(y)

	for label in labels:
		print("label: {} num: {}".format(label, np.sum(np.array(y)==label)))


def reduce_to(x_raw, y_raw, index_raw, n_data, data_percs, dpi):
	"""
	reduce to smaller but equal number of samples and classes
	"""

	# to numpy
	y_raw = np.array(y_raw)
	index_raw = np.array(index_raw)

	# get labels
	labels = np.unique(y_raw)

	# init
	n = int(data_percs[dpi] * n_data * len(labels))
	x = np.empty(shape=((n,) + x_raw.shape[1:]), dtype=x_raw.dtype)
	y = np.empty(shape=(n,), dtype=y_raw.dtype)
	index = np.empty(shape=(n,), dtype=index_raw.dtype)

	# number of examples per label
	n_label = int(data_percs[dpi] * n_data)

	# splitting
	for i, label in enumerate(labels):
		x[i*n_label:i*n_label+n_label] = x_raw[y_raw==label][:n_label]
		y[i*n_label:i*n_label+n_label] = y_raw[y_raw==label][:n_label]
		index[i*n_label:i*n_label+n_label] = index_raw[y_raw==label][:n_label]

	return x, y, index


if __name__ == '__main__':
	"""
	main function of audio dataset
	"""

	# path to whole dataset
	dataset_path = './ignore/speech_commands_v0.01/'

	# path to training, test and eval set
	data_paths = ['./ignore/train/', './ignore/test/', './ignore/eval/']

	# percent of data splitting [train, test], leftover is eval
	data_percs = np.array([0.8, 0.1, 0.1])

	# version number
	version_nr = 2

	# num examples per class for ml 
	#n_examples, n_data = 12, 10
	#n_examples, n_data = 70, 50
	#n_examples, n_data = 550, 500
	n_examples, n_data = 2200, 2000

	# plot path
	plot_path = './ignore/plots/features/n{}/'.format(n_examples)

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
	params = {'n_examples':n_examples, 'data_percs':data_percs, 'fs':fs, 'N':N, 'hop':hop, 'n_filter_bands':n_filter_bands, 'n_ceps_coeff':n_ceps_coeff}

	# mfcc info
	info = "n_examples={} with data split {}, fs={}, mfcc: N={} is t={}, hop={} is t={}, n_f-bands={}, n_ceps_coeff={}".format(n_examples, data_percs, fs, N, N/fs, hop, hop/fs, n_filter_bands, n_ceps_coeff)
	
	# some prints
	print(info)
	print("params: ", params)

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
		x_raw, y_raw, index_raw = extract_mfcc_data(wavs, params, frame_size=32, ext=ext, plot_path=None)

		# print label stats
		label_stats(y_raw)

		# reduce to same amount of labels
		x, y, index = reduce_to(x_raw, y_raw, index_raw, n_data, data_percs, dpi)

		# stats
		print("reduces: ")
		label_stats(y)

		# set file name
		file_name = '{}mfcc_data_{}_n-{}_c-{}_v{}.npz'.format(data_path, ext, n_data, len(sel_labels), version_nr)

		# save mfcc data file
		np.savez(file_name, x=x, y=y, index=index, info=info, params=params)

		# print
		print("--save data to: ", file_name)



	# show plots
	#plt.show()


