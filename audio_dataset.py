"""
audio datasets set creation for kws game
process of the speech command dataset and extraction to MFCC features
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import librosa
import yaml
import os

from glob import glob
from shutil import copyfile

# my stuff
from feature_extraction import *
from common import create_folder
from plots import *


def	create_datasets(n_examples, dataset_path, data_paths, data_percs):
	"""
	copy wav files from dataset_path to data_paths with splitting
	"""

	# return if data paths exist
	if os.path.isdir(data_paths[0]):
		return

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

		# calculate split numbers in train, test, eval
		n_split = (len(wavs) * np.array(data_percs)).astype(int)

		# determine position to split
		n_split_pos = np.cumsum(n_split)

		# print some info
		print("label: [{}]\tn_split: [{}]\ttotal:[{}]".format(label, n_split, np.sum(n_split)))

		# actual path
		p = 0

		# run through each path
		for i, wav in enumerate(wavs):

			# split in new path
			if i >= n_split_pos[p]:
				p += 1

			# stop if out of range (happens at rounding errors)
			if p >= len(data_paths):
				break

			# copy files to folder
			copyfile(wav, data_paths[p] + label + str(i) + '.wav')

	return labels


def wav_pre_processing(wav, fs, min_samples):
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

	return x_raw


def extract_mfcc_data(wavs, params, n_examples, set_name=None, min_samples=16000, plot_path_mfcc=None, plot_path_z_score=None, enable_plot=False):
	"""
	extract mfcc data from wav-files
	wavs must be in a 2D-array [[wavs1], [wavs2]]
	"""

	# windowing samples
	N, hop = int(params['N_s'] * params['fs']), int(params['hop_s'] * params['fs'])

	# init mfcc_data: [n x m x l] n samples, m features, l frames
	mfcc_data = np.empty(shape=(0, params['feature_size'], params['frame_size']), dtype=np.float64)
	
	# some lists
	z_score_list, broken_file_list, label_data, index_data = np.array([]), [], [], []

	# create feature extractor
	feature_extractor = FeatureExtractor(fs=params['fs'], N=N, hop=hop, n_filter_bands=params['n_filter_bands'], n_ceps_coeff=params['n_ceps_coeff'], frame_size=params['frame_size'])

	# extract class wavs
	for class_wavs in wavs:

		# run through each example in class wavs
		for i, wav in enumerate(class_wavs):
		
			# extract file namings
			file_name, file_index, label = file_naming_extraction(wav)

			# load and pre-process audio
			x = wav_pre_processing(wav, params['fs'], min_samples)

			# print some info
			print("wav: [{}] with label: [{}], samples=[{}], time=[{}]s".format(wav, label, len(x), len(x)/params['fs']))

			# extract feature vectors [m x l]
			mfcc, bon_pos = feature_extractor.extract_mfcc39(x)

			# onset analysis if necessary
			#onset_analysis(x, params, N, hop)

			# damaged file things
			z_score, z_damaged = detect_damaged_file(mfcc)
			z_score_list = np.append(z_score_list, z_score)

			# plot mfcc features
			plot_mfcc_profile(x, params['fs'], N, hop, mfcc, onsets=None, bon_pos=bon_pos, mient=None, minreg=None, frame_size=params['frame_size'], plot_path=plot_path_mfcc, name=label + str(file_index) + '_' + set_name, enable_plot=enable_plot)

			# handle damaged files
			if z_damaged:
				print("--*file probably broken!")
				broken_file_list.append(file_name)
				continue

			# add to mfcc_data container
			mfcc_data = np.vstack((mfcc_data, mfcc[np.newaxis, :, :]))
			label_data.append(label)
			index_data.append(label + file_index)

			# stop if desired examples are reached
			if i >= n_examples:
				break

	# broken file info
	plot_damaged_file_score(z_score_list, plot_path=plot_path_z_score, name='z_score_n-{}_{}'.format(n_examples, set_name), enable_plot=enable_plot)
	print("\n --broken file list: \n{}\nwith length: {}".format(broken_file_list, len(broken_file_list)))

	return mfcc_data, label_data, index_data


def file_naming_extraction(wav):
	"""
	extracts the file name, index and label of example
	"""

	# extract filename
	file_name = re.findall(r'[\w+ 0-9]+\.wav', wav)[0]

	# extract file index from filename
	file_index = re.sub(r'[a-z A-Z]|(\.wav)', '', file_name)

	# extract label from filename
	label = re.sub(r'([0-9]+\.wav)', '', file_name)

	return file_name, file_index, label


def onset_analysis(x, params, N, hop):
	"""
	onset analysis - only for evaluation
	"""

	# calc onsets
	onsets = calc_onsets(x, params['fs'], N=N, hop=hop, adapt_frames=5, adapt_alpha=0.1, adapt_beta=1)
	#onsets = calc_onsets(x, params['fs'], N=N, hop=hop, adapt_frames=5, adapt_alpha=0.05, adapt_beta=0.9)

	# find best onset
	#best_onset, bon_pos = find_best_onset(onsets, frame_size=frame_size, pre_frames=1)
	#mient = find_min_energy_time(mfcc, params['fs'], hop)
	#minreg, bon_pos = find_min_energy_region(mfcc, params['fs'], hop, frame_size=params['frame_size'], randomize=True)


def detect_damaged_file(mfcc, z_lim=60):
	"""
	detect if file is damaged
	"""

	# calculate damaged score of energy deltas
	z_est = np.sum(mfcc[37:39, :])

	# return score and damaged indicator
	return z_est, z_est > z_lim


def label_stats(y):
	"""
	label statistics
	"""

	labels = np.unique(y)

	for label in labels:
		print("label: {} num: {}".format(label, np.sum(np.array(y)==label)))


def reduce_to(x_raw, y_raw, index_raw, n_data, data_percs, dpi):
	"""
	reduce to smaller but equal number of samples and classes - not used anymore
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

	# yaml config file
	cfg = yaml.safe_load(open("./config.yaml"))

	# create folder
	create_folder([p + cfg['audio_dataset']['wav_folder'] for p in cfg['audio_dataset']['data_paths']] +  [cfg['audio_dataset']['plot_paths']['mfcc']] + [cfg['audio_dataset']['plot_paths']['z_score']])

	# status message
	print("\n--create datasets\nexamples per class: [{}], saved at paths: {} with splits: {}\n".format(cfg['audio_dataset']['n_examples'], cfg['audio_dataset']['data_paths'], cfg['audio_dataset']['data_percs']))

	# copy wav files to path
	labels = create_datasets(cfg['audio_dataset']['n_examples'], cfg['audio_dataset']['dataset_path'], [p + cfg['audio_dataset']['wav_folder'] for p in cfg['audio_dataset']['data_paths']], cfg['audio_dataset']['data_percs'])


	# --
	# extract mfcc features

	# print params
	print("params: ", cfg['feature_params'])

	# for all data paths (train, test, eval)
	for dpi, data_path in enumerate(cfg['audio_dataset']['data_paths']):

		print("\ndata_path: ", data_path)

		# extract set name e.g train
		set_name = re.sub(r'/', '', re.findall(r'[\w+ 0-9]+/', data_path)[-1])

		# init wavs
		all_wavs = []

		# get all wavs from selected labels
		for l in cfg['audio_dataset']['sel_labels']:

			# wav regex
			wav_name_re = '*' + l + '[0-9]*.wav'

			# get wavs
			wavs = glob(data_path + cfg['audio_dataset']['wav_folder'] + wav_name_re)

			# add to all wavs
			all_wavs.append(wavs)

			# check length of labels
			print("label: {} with n: {}".format(l, len(wavs)))

		# extract data
		x, y, index = extract_mfcc_data(all_wavs, cfg['feature_params'], int(cfg['audio_dataset']['n_examples']*cfg['audio_dataset']['data_percs'][dpi]), set_name=set_name, plot_path_mfcc=cfg['audio_dataset']['plot_paths']['mfcc'], plot_path_z_score=cfg['audio_dataset']['plot_paths']['z_score'], enable_plot=cfg['audio_dataset']['enable_plot'])

		# print label stats
		label_stats(y)

		# set file name
		file_name = '{}mfcc_data_{}_n-{}_c-{}_v{}.npz'.format(data_path, set_name, cfg['audio_dataset']['n_examples'], len(cfg['audio_dataset']['sel_labels']), cfg['audio_dataset']['version_nr'])

		# save mfcc data file
		np.savez(file_name, x=x, y=y, index=index, params=cfg['feature_params'])

		# print
		print("--save data to: ", file_name)


