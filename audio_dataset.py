"""
audio datasets set creation for kws game
process of the speech command dataset and extraction to MFCC features
"""

import re
import numpy as np
import librosa
import soundfile

from glob import glob
from shutil import copyfile

# my stuff
from feature_extraction import FeatureExtractor, calc_onsets
from plots import plot_mfcc_profile, plot_damaged_file_score, plot_onsets, plot_waveform
from common import check_folders_existance, create_folder


class AudioDataset():
	"""
	audio dataset interface with some useful functions
	"""

	def __init__(self, dataset_cfg):

		# params
		self.dataset_cfg = dataset_cfg

		# variables
		self.labels = []
		self.set_names = []
		self.set_audio_files = []

		# parameter path
		self.param_path = 'v{}_c-{}_n-{}/'.format(self.dataset_cfg['version_nr'], len(self.dataset_cfg['sel_labels']), self.dataset_cfg['n_examples'])

		# wav folders
		self.wav_folders = [p + self.dataset_cfg['wav_folder'] for p in list(self.dataset_cfg['data_paths'].values())]

		# feature folders
		self.feature_folders = [p + self.param_path for p in list(self.dataset_cfg['data_paths'].values())]

		# feature files
		self.feature_files = [p + '{}.npz'.format(self.dataset_cfg['feature_file_name']) for p in self.feature_folders]


	def create_sets(self):
		"""
		create set structure (e.g. train, eval, test)
		"""
		pass


	def extract_features(self):
		"""
		extract features from dataset
		"""
		pass


	def file_naming_extraction(self, audio_file):
		"""
		extracts the file name, index and label of a file
		convention to filename: e.g. label123.wav
		"""

		# extract filename
		file_name = re.findall(r'[\w+ 0-9]+' + re.escape(self.dataset_cfg['file_ext']), audio_file)[0]

		# extract file index from filename
		file_index = re.sub(r'[a-z A-Z]|(' + re.escape(self.dataset_cfg['file_ext']) + r')', '', file_name)

		# extract label from filename
		label = re.sub(r'([0-9]+' + re.escape(self.dataset_cfg['file_ext']) + r')', '', file_name)

		return file_name, file_index, label


	def get_audiofiles(self):
		"""
		get audiofiles from datapaths
		return [[data_path1], [data_path2], ...]
		"""

		print("\n--get audiofiles:")

		# for all data paths (train, test, eval)
		for dpi, data_path in enumerate(list(self.dataset_cfg['data_paths'].values())):

			# determine set name
			set_name = re.sub(r'/', '', re.findall(r'[\w+ 0-9]+/', data_path)[-1])
			print("set name: ", set_name)
			self.set_names.append(set_name)

			# init set files
			set_files = []

			# get all wavs from selected labels
			for l in self.dataset_cfg['sel_labels']:

				# regex
				file_name_re = '*' + l + '[0-9]*' + self.dataset_cfg['file_ext']

				# get wavs
				label_files = glob(data_path + self.dataset_cfg['wav_folder'] + file_name_re)
				set_files.append(label_files)

				# check length of label files
				print("overall stat of label: [{}]\tnum: [{}]".format(l, len(label_files)))

				# check label num
				if len(label_files) < int(self.dataset_cfg['n_examples'] * self.dataset_cfg['split_percs'][dpi]):
					print("***[get_audiofiles] labels are less than n_examples, recreate dataset and check files")
					import sys
					sys.exit()

			# update set audio files
			self.set_audio_files.append(set_files)


	def label_stats(self, y):
		"""
		label statistics
		"""

		print("\nlabel stats:")

		# get labels
		labels = np.unique(y)

		# print label stats
		for label in labels:
			label_num = np.sum(np.array(y)==label)
			print("label: [{}]\tnum: [{}]".format(label, label_num))



class SpeechCommandsDataset(AudioDataset):
	"""
	Speech Commands Dataset extraction and set creation
	"""

	def __init__(self, dataset_cfg, feature_params, verbose=False):

		# parent init
		super().__init__(dataset_cfg)

		# arguments
		self.feature_params = feature_params
		self.verbose = verbose

		# feature extractor
		self.feature_extractor = FeatureExtractor(feature_params=self.feature_params)

		# short vars
		self.N = self.feature_extractor.N
		self.hop = self.feature_extractor.hop

		# minimum amount of samples per example
		self.min_samples = 16000

		# recreate
		if self.dataset_cfg['recreate'] or not check_folders_existance(self.wav_folders + self.feature_folders, empty_check=True):

			# create folder structure
			create_folder(self.wav_folders + self.feature_folders + list(self.dataset_cfg['plot_paths'].values()))

			# create sets (specific to dataset)
			self.create_sets()

		# get audio files from sets
		self.get_audiofiles()


	def create_sets(self):
		"""
		copy wav files from dataset path to wav folders with splitting
		"""

		# get all class directories except the ones starting with _
		class_dirs = glob(self.dataset_cfg['dataset_path'] + '[!_]*/')

		# run through all class directories
		for class_dir in class_dirs:

			# extract label
			label = class_dir.split('/')[-2]

			# append to label list
			self.labels.append(label)

			# get all .wav files
			wavs = glob(class_dir + '*' + self.dataset_cfg['file_ext'])

			# calculate split numbers in train, test, eval and split position
			n_split = (len(wavs) * np.array(self.dataset_cfg['split_percs'])).astype(int)
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
				if p >= len(self.wav_folders):
					break

				# copy files to folder
				copyfile(wav, self.wav_folders[p] + label + str(i) + self.dataset_cfg['file_ext'])


	def extract_features(self):
		"""
		extract mfcc features and save them
		"""

		print("\n--feature extraction:")

		for i, (set_name, wavs) in enumerate(zip(self.set_names, self.set_audio_files)):

			print("{}) extract set: {} with label num: {}".format(i, set_name, len(wavs)))

			# examples with splits
			n_examples = int(self.dataset_cfg['n_examples']*self.dataset_cfg['split_percs'][i])

			# extract data
			x, y, index = self.extract_mfcc_data(wavs=wavs, n_examples=n_examples, set_name=set_name)

			# print label stats
			self.label_stats(y)

			# save mfcc data file
			np.savez(self.feature_files[i], x=x, y=y, index=index, params=cfg['feature_params'])
			print("--save data to: ", self.feature_files[i])


	def extract_mfcc_data(self, wavs, n_examples, set_name=None):
		"""
		extract mfcc data from wav-files
		wavs must be in a 2D-array [[wavs_class1], [wavs_class2]] so that n_examples will work properly
		"""

		# mfcc_data: [n x m x l], labels and index
		mfcc_data, label_data, index_data = np.empty(shape=(0, self.feature_params['feature_size'], self.feature_params['frame_size']), dtype=np.float64), [], []

		# some lists
		z_score_list, broken_file_list,  = np.array([]), []

		# extract class wavs
		for class_wavs in wavs:

			# number of class examples
			num_class_examples = 0

			# run through each example in class wavs
			for wav in class_wavs:
			
				# extract file namings
				file_name, file_index, label = self.file_naming_extraction(wav)

				# load and pre-process audio
				x = self.wav_pre_processing(wav)

				# print some info
				if self.verbose:
					print("wav: [{}] with label: [{}], samples=[{}], time=[{}]s".format(wav, label, len(x), len(x)/self.feature_params['fs']))

				# extract feature vectors [m x l]
				mfcc, bon_pos = self.feature_extractor.extract_mfcc39(x, normalize=self.feature_params['norm_features'], reduce_to_best_onset=False)

				# damaged file things
				z_score, z_damaged = self.detect_damaged_file(mfcc)
				z_score_list = np.append(z_score_list, z_score)

				# plot mfcc features
				plot_mfcc_profile(x, self.feature_params['fs'], self.feature_extractor.N, self.feature_extractor.hop, mfcc, onsets=None, bon_pos=bon_pos, mient=None, minreg=None, frame_size=self.feature_params['frame_size'], plot_path=self.dataset_cfg['plot_paths']['mfcc'], name=label + str(file_index) + '_' + set_name, enable_plot=self.dataset_cfg['enable_plot'])

				# handle damaged files
				if z_damaged:
					if self.verbose:
						print("--*file probably broken!")
					broken_file_list.append(file_name)
					continue

				# add to mfcc_data container
				mfcc_data = np.vstack((mfcc_data, mfcc[np.newaxis, :, bon_pos:bon_pos+self.feature_params['frame_size']]))
				label_data.append(label)
				index_data.append(label + file_index)

				# update number of examples per class
				num_class_examples += 1

				# stop if desired examples are reached
				if num_class_examples >= n_examples:
					break

		# broken file info
		plot_damaged_file_score(z_score_list, plot_path=self.dataset_cfg['plot_paths']['z_score'], name='z_score_n-{}_{}'.format(n_examples, set_name), enable_plot=self.dataset_cfg['enable_plot'])
		
		if self.verbose:
			print("\nbroken file list: [{}] with length: [{}]".format(broken_file_list, len(broken_file_list)))

		return mfcc_data, label_data, index_data


	def wav_pre_processing(self, wav):
		"""
		Audio pre-processing stage
		"""

		# read audio from file
		x_raw, fs = librosa.load(wav, sr=self.feature_params['fs'])

		# check sample lengths
		if len(x_raw) < self.min_samples:

			# print warning
			if self.verbose:
				print("lengths is less than 1s, append with zeros for:")

			# append with zeros
			x_raw = np.append(x_raw, np.zeros(self.min_samples - len(x_raw)))

		return x_raw


	def detect_damaged_file(self, mfcc, z_lim=60):
		"""
		detect if file is damaged
		"""

		# calculate damaged score of energy deltas
		z_est = np.sum(mfcc[37:39, :])

		# return score and damaged indicator
		return z_est, z_est > z_lim



class MyRecordingsDataset(SpeechCommandsDataset):
	"""
	Speech Commands Dataset extraction and set creation
	"""

	def __init__(self, dataset_cfg, feature_params, verbose=False):

		# parent init
		super().__init__(dataset_cfg, feature_params, verbose)


	def create_sets(self):
		"""
		cut and copy recorded wavs
		"""

		print("wav: ", self.wav_folders)

		# get all .wav files
		raw_wavs = glob(self.dataset_cfg['dataset_path'] + '*' + self.dataset_cfg['file_ext'])

		# get all wav files and save them
		for i, wav in enumerate(raw_wavs):

			print("wav: ", wav)

			# filename extraction
			file_name, file_index, label = self.file_naming_extraction(wav)

			# append to label list
			self.labels.append(label)

			# read audio from file
			x, _ = librosa.load(wav, sr=self.feature_params['fs'])

			# calc onsets
			onsets = calc_onsets(x, self.feature_params['fs'], N=self.N, hop=self.hop, adapt_frames=5, adapt_alpha=0.09, adapt_beta=0.8)
			onsets = self.clean_onsets(onsets)

			# cut examples to one second
			x_cut = self.cut_signal(x, onsets, time=1, alpha=0.4)

			# plot onsets
			plot_onsets(x, self.feature_params['fs'], self.N, self.hop, onsets, title=label, plot_path=self.dataset_cfg['plot_paths']['onsets'], name='onsets_{}'.format(label))

			for j, xj in enumerate(x_cut):

				# plot
				plot_waveform(xj, self.feature_params['fs'], title='{}-{}'.format(label, j), plot_path=self.dataset_cfg['plot_paths']['waveform'], name='example_{}-{}'.format(label, j))

				# save file
				soundfile.write('{}{}{}.wav'.format(self.wav_folders[0], label, j), xj, self.feature_params['fs'], subtype=None, endian=None, format=None, closefd=True)


	def clean_onsets(self, onsets):
		"""
		clean onsets, so that only one per examples exists
		"""

		# analytical frame size
		analytic_frame_size = self.feature_params['frame_size'] * 5

		# save old onsets
		onsets = onsets.copy()

		# init
		clean_onsets = np.zeros(onsets.shape)

		# setup primitive onset filter
		onset_filter = np.zeros(analytic_frame_size)
		onset_filter[0] = 1

		# onset filtering
		for i, onset in enumerate(onsets):

			# stop at end
			if i > len(onsets) - analytic_frame_size // 2:
				break

			# clean
			if onset:
				onsets[i:i+analytic_frame_size] = onset_filter
				clean_onsets[i] = 1

		return clean_onsets


	def cut_signal(self, x, onsets, time=1, alpha=0.5):
		"""
		cut signal at onsets to a specific time interval
		"""

		# init cut signal
		x_cut = np.zeros((int(np.sum(onsets)), int(self.feature_params['fs'] * time)))

		# post factor
		beta = 1 - alpha

		# amount of samples 
		num_samples = self.feature_params['fs'] / time

		# pre and post samples
		pre = int(alpha * num_samples)
		post = int(beta * num_samples)

		# onset index
		oi = 0

		for i, onset in enumerate(onsets):

			if onset:
				x_cut[oi, :] = x[i*self.hop-pre:i*self.hop+post]
				oi += 1

		return x_cut


if __name__ == '__main__':
	"""
	main function of audio dataset
	"""

	import yaml

	# yaml config file
	cfg = yaml.safe_load(open("./config.yaml"))

	# audioset init
	audio_dataset = SpeechCommandsDataset(cfg['datasets']['speech_commands'], feature_params=cfg['feature_params'], verbose=False)
	#audio_dataset = MyRecordingsDataset(cfg['datasets']['my_recordings'], feature_params=cfg['feature_params'], verbose=False)

	# extract and save features
	audio_dataset.extract_features()






