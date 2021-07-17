"""
audio datasets set creation for kws game
process of the speech command dataset and extraction to MFCC features
"""

import re
import os
import numpy as np
import librosa
import soundfile

from glob import glob
from shutil import copyfile

# my stuff
from feature_extraction import FeatureExtractor, calc_onsets
from plots import plot_mfcc_profile, plot_damaged_file_score, plot_onsets, plot_waveform, plot_histogram
from common import check_folders_existance, check_files_existance, create_folder, delete_files_in_path
from skimage.util.shape import view_as_windows


class AudioDataset():
  """
  audio dataset interface with some useful functions
  """

  def __init__(self, cfg_dataset, feature_params, root_path='./'):

    # params
    self.cfg_dataset = cfg_dataset
    self.feature_params = feature_params
    self.root_path = root_path

    # data sizes
    self.channel_size = 1 if not self.feature_params['use_channels'] or not self.feature_params['use_mfcc_features'] else int(self.feature_params['use_cepstral_features']) + int(self.feature_params['use_delta_features']) +  int(self.feature_params['use_double_delta_features'])
    self.feature_size = (self.feature_params['n_ceps_coeff'] + int(self.feature_params['use_energy_features'])) * int(self.feature_params['use_cepstral_features']) + (self.feature_params['n_ceps_coeff'] + int(self.feature_params['use_energy_features'])) * int(self.feature_params['use_delta_features']) + (self.feature_params['n_ceps_coeff'] + int(self.feature_params['use_energy_features'])) * int(self.feature_params['use_double_delta_features']) if not self.feature_params['use_channels'] else (self.feature_params['n_ceps_coeff'] + int(self.feature_params['use_energy_features']))
    self.frame_size = self.feature_params['frame_size']
    self.raw_frame_size = int(self.feature_params['frame_size_s'] * self.feature_params['fs'])

    # feature extractor
    self.feature_extractor = FeatureExtractor(feature_params=self.feature_params)
    self.N = self.feature_extractor.N
    self.hop = self.feature_extractor.hop

    # variables
    self.sel_labels = self.cfg_dataset['sel_labels']
    self.all_labels = self.cfg_dataset['all_labels']

    # add special labels
    if self.cfg_dataset['add_noise']: self.sel_labels, self.all_labels = self.sel_labels + [self.cfg_dataset['noise_label']], self.all_labels + [self.cfg_dataset['noise_label']]
    if self.cfg_dataset['add_mixed']: self.sel_labels, self.all_labels = self.sel_labels + [self.cfg_dataset['mixed_label']], self.all_labels + [self.cfg_dataset['mixed_label']]

    # mixed labels
    self.mixed_labels = [i for i in self.cfg_dataset['all_labels'] if i not in self.cfg_dataset['sel_labels']] if self.cfg_dataset['add_mixed'] else None

    # init label file dicts
    self.all_label_file_dict = {l: {} for l in self.all_labels}
    self.sel_label_file_dict = {set_name: {} for set_name in self.cfg_dataset['set_folders'].keys()}
    self.set_annotation_file_dict = {}

    # dataset infos
    self.dataset_path = self.root_path + self.cfg_dataset['dataset_path']
    self.dataset_name = self.dataset_path.split('/')[-2]
    self.dataset_version = re.findall(r'v[0-9].[0-9]+', self.dataset_name)

    # extraction path
    self.extraction_path = self.root_path + self.cfg_dataset['extraction_path'] + self.dataset_name + '/'

    # plot paths
    self.plot_paths = {k: self.extraction_path + v for k, v in self.cfg_dataset['plot_paths'].items()}

    # parameter path
    self.param_path = 'v{}_c{}n{}m{}_n-{}_r{}-{}'.format(self.cfg_dataset['version_nr'], len(self.sel_labels), int(self.cfg_dataset['add_noise']), int(self.cfg_dataset['add_mixed']), self.cfg_dataset['n_examples'], int(self.cfg_dataset['rand_best_onset']), int(self.cfg_dataset['rand_delta_percs'] * 100))
    self.param_path += '_mfcc{}-{}_c{}d{}d{}e{}_norm{}_f-{}x{}x{}/'.format(self.feature_params['n_filter_bands'], self.feature_params['n_ceps_coeff'], int(self.feature_params['use_cepstral_features']), int(self.feature_params['use_delta_features']), int(self.feature_params['use_double_delta_features']), int(self.feature_params['use_energy_features']), int(self.feature_params['norm_features']), self.channel_size, self.feature_size, self.frame_size) if self.feature_params['use_mfcc_features'] else '_raw/'

    # folders
    self.wav_folder_dict = {k: self.extraction_path + v + self.cfg_dataset['wav_folder'] for k, v in self.cfg_dataset['set_folders'].items()}
    self.annotation_folder_dict = {k: self.extraction_path + v + self.cfg_dataset['annotation_folder'] for k, v in self.cfg_dataset['set_folders'].items()}
    self.feature_folder_dict = {k: self.extraction_path + v + self.param_path for k, v in self.cfg_dataset['set_folders'].items()}

    # feature files
    self.feature_file_dict = {k: v + '{}.npz'.format(self.cfg_dataset['mfcc_feature_file_name']) for k, v in self.feature_folder_dict.items()} if self.feature_params['use_mfcc_features'] else {k: v + '{}.npz'.format(self.cfg_dataset['raw_feature_file_name']) for k, v in self.feature_folder_dict.items()}


    # statistics dict
    self.stats_dict = {'all_extracted_wavs': [], 'sample_num': [], 'energy': [], 'damaged_score': []}

    # info file dict
    self.info_file_dict = {'damaged': [], 'short': [], 'weak': [], 'strong': []}


    # ignore list for damaged files (was only used for speech commands v0.01, v0.02 is good enough)
    try:
      with open(self.plot_paths['damaged_files'] + 'damaged_file_list.txt') as f:
        self.ignore_file_list_re = '(' + ')|('.join([line.strip() for line in f]) +')'
    except: 
      self.ignore_file_list_re = []


  def create_sets(self):
    """
    create sets
    """
    pass


  def extract_features(self):
    """
    extract features from dataset
    """
    pass


  def reset_statistics(self):
    """
    reset statistics
    """
    self.stats_dict, self.info_file_dict = {k: [] for k in self.stats_dict.keys()}, {k: [] for k in self.info_file_dict.keys()}


  def get_audiofiles(self):
    """
    get audiofiles from set folder paths
    """

    # update label files dict
    [(print('\nlabel files in set wav folder: ', wav_folder), [self.all_label_file_dict[l].update({set_name: self.get_label_files_from_folder(l, wav_folder, set_name)}) for l in self.all_labels if l != self.cfg_dataset['mixed_label']]) for set_name, wav_folder in self.wav_folder_dict.items()]

    # add all labels to sets, except for mixed label
    [[self.sel_label_file_dict[set_name].update({l: self.all_label_file_dict[l][set_name]}) for l in self.sel_labels if l != self.cfg_dataset['mixed_label']] for set_name, _ in self.wav_folder_dict.items()]

    # add mixed labels if there exists any
    if self.mixed_labels is not None:

      # create mixed label
      all_mixed_label_file_dict = {set_name: {l: self.all_label_file_dict[l][set_name] for l in self.mixed_labels} for set_name, _ in self.wav_folder_dict.items()}

      # minimum number of mixed files
      min_num_mixed_dict = {set_name: np.min([len(wavs) for l, wavs in label_dict.items()]) for set_name, label_dict in all_mixed_label_file_dict.items()}

      # mixed label files sorted after each other
      mixed_label_file_dict = {set_name: np.concatenate([[all_mixed_label_file_dict[set_name][l][i] for l in self.mixed_labels] for i in range(min_num_mixed_dict[set_name])]).tolist() for set_name, _ in self.wav_folder_dict.items()}

      # add mixed label
      [self.sel_label_file_dict[set_name].update({self.cfg_dataset['mixed_label']: mixed_label_file_dict[set_name]}) for set_name, _ in self.wav_folder_dict.items()]


  def get_label_files_from_folder(self, label, wav_folder, set_name):
    """
    get label files from wav folders
    """

    # regex
    file_name_re = '*' + label + '[0-9]*' + self.cfg_dataset['file_ext']

    # get wavs
    label_files = sorted(glob(wav_folder + file_name_re))

    # check length of label files
    print("overall stat of label: [{}]\tnum: [{}]".format(label, len(label_files)))

    # check label num
    if len(label_files) < int(self.cfg_dataset['n_examples'] * self.cfg_dataset['split_percs'][set_name]):
      print("***[audio set] labels are less than n_examples, recreate dataset and check files")
      import sys
      sys.exit()

    return label_files


  def extract_wav_examples(self, set_name, n_examples, from_selected_labels=True):
    """
    extract some wavs for evaluation
    """

    # init info dicts
    info_dicts = []

    # each label
    for label, wavs in self.sel_label_file_dict[set_name].items() if from_selected_labels else self.all_label_file_dict.items():

      # add info dicts
      info_dicts += [{'y': label, 'x': np.squeeze(x), 'bon_pos': bon_pos} for x, bon_pos in [self.feature_extractor.extract_raw(self.wav_pre_processing(wav)[0], reduce_to_best_onset=False) for i, wav in zip(range(n_examples), wavs if from_selected_labels else (wavs[set_name] if len(wavs.keys()) else []))]]

    return info_dicts


  def get_annotation_files(self):
    """
    get annotation files
    """

    # for all data paths
    for set_name, annotation_folder in self.annotation_folder_dict.items():

      # info
      print("\nset annotation folder: {}".format(annotation_folder))

      # init set files
      set_files = []

      # get all wavs from selected labels
      for l in self.sel_labels:

        # regex
        file_name_re = '*' + l + '[0-9]*' + '.TextGrid'

        # get wavs
        label_files = glob(annotation_folder + file_name_re)
        set_files.append(label_files)

        # check length of label files
        print("overall stat of anno: [{}]\tnum: [{}]".format(l, len(label_files)))

      # update set file dict
      self.set_annotation_file_dict.update({set_name: set_files})


  def label_stats(self, y):
    """
    label statistics
    """
    print("\nlabel stats:"), [print("label: [{}]\tnum: [{}]".format(label, np.sum(np.array(y)==label))) for label in np.unique(y)]


  def add_noise_to_dataset(self, x, y, z, n_examples):
    """
    add noise data
    """

    # create noise
    n = np.random.rand(n_examples, x.shape[1], x.shape[2], x.shape[3])

    # update
    x = np.vstack((x, n))
    y = y + [self.cfg_dataset['noise_label'] for i in range(n_examples)]
    z = z + [self.cfg_dataset['noise_label'] + str(i) for i in range(n_examples)]

    return x, y, z


  def wav_pre_processing(self, wav):
    """
    audio pre processing, check if file is okay and clean up a bit, no normalization
    """

    # check ignore file list
    if len(self.ignore_file_list_re):
      ignore = re.findall(self.ignore_file_list_re, wav)
      if len(ignore):
        print("ignore wav: ", wav)
        return 0, True

    # read audio from file
    x_raw, fs = librosa.load(wav, sr=self.feature_params['fs'])

    # energy of raw audio
    e = (x_raw @ x_raw.T) / len(x_raw)

    # save some stats
    self.stats_dict['all_extracted_wavs'].append(wav)
    self.stats_dict['sample_num'].append(len(x_raw))
    self.stats_dict['energy'].append(e)

    # too short flag
    is_too_short = len(x_raw) < self.cfg_dataset['sample_num_mininmal']
    is_too_weak = e < 1e-6
    is_too_strong = e > 0.2

    # too short
    if is_too_short: self.info_file_dict['short'].append((wav, len(x_raw)))
    if is_too_weak: self.info_file_dict['weak'].append((wav, e))
    if is_too_strong: self.info_file_dict['strong'].append((wav, e))

    # check sample lengths
    if len(x_raw) < self.cfg_dataset['sample_num_normal']:

      # print warning
      if self.cfg_dataset['verbose']: print("lengths is less than 1s, append with zeros for:")

      # append with zeros
      x_raw = np.append(x_raw, np.zeros(self.cfg_dataset['sample_num_normal'] - len(x_raw)))

    # what makes the wav useless
    wav_is_useless = is_too_weak if self.cfg_dataset['filter_damaged_files'] and not self.cfg_dataset['noise_label'] in wav else False

    return x_raw, wav_is_useless


  def analyze_dataset_extraction(self, calculate_overall_stats=False):
    """
    analyze damaged files
    """

    # some info prints
    print("\n--analyze dataset extraction from ", self.__class__.__name__)

    # plot statistics if available
    if any(self.stats_dict.keys()): self.plot_statistics(file_ext='_extracted_n-{}'.format(self.cfg_dataset['n_examples']))

    # get speaker infos
    all_speakers_files, all_speakers = self.get_speaker_infos(self.stats_dict['all_extracted_wavs'])

    # print info
    print("selected labels: ", self.sel_labels), print("number of extracted files: ", len(self.stats_dict['all_extracted_wavs'])), print("speakers: ", all_speakers), print("number of speakers: ", len(all_speakers))

    # save some energy
    if not calculate_overall_stats: return

    # info message
    print("\n--compute overall stats for ", self.__class__.__name__)

    # all wavs
    all_wavs = np.concatenate(np.concatenate(np.array([list(set_dict.values()) for set_dict in [self.all_label_file_dict[l] for l in self.all_labels if l != self.cfg_dataset['noise_label'] and l != self.cfg_dataset['mixed_label']]], dtype='object')))

    # get speaker infos
    all_speakers_files, all_speakers = self.get_speaker_infos(all_wavs)

    # print info
    print("all labels: ", self.all_labels), print("num of labels: ", len(self.all_label_file_dict.keys()))
    print("number of files: ", len(all_speakers_files)), print("speakers: ", all_speakers), print("number of speakers: ", len(all_speakers))

    # reset statistics
    self.reset_statistics()

    # compute all wavs in dataset
    for wav in all_wavs: self.wav_pre_processing(wav)

    # plot overall stats
    self.plot_statistics(file_ext='_overall')


  def get_speaker_infos(self, wavs):
    """
    speaker infos of all files
    """

    # extract speaker info
    all_speakers_files = [re.sub(r'^.*(?=--)|(_nohash_[0-9]+.wav)|(--)', '', i) for i in wavs]

    # filter noise files
    all_speakers_files = [i for i in all_speakers_files if '_noise' not in i]
    
    # get all unique speakers
    all_speakers = np.unique(all_speakers_files)

    return all_speakers_files, all_speakers


  def plot_statistics(self, plot_path=None, file_ext=''):
    """
    plot statistics
    """

    if plot_path is None: plot_path = self.plot_paths['stats']

    # histogram
    #if len(self.stats_dict['energy']): plot_histogram(self.stats_dict['energy'], bins=np.logspace(np.log10(0.0001),np.log10(10000), 50), y_log_scale=True, x_log_scale=True, x_label='energy values', y_label='counts', plot_path=plot_path, name='energy_hist' + file_ext)
    if len(self.stats_dict['energy']): plot_histogram(self.stats_dict['energy'], bins=np.logspace(np.log10(1e-6),np.log10(1), 50), y_log_scale=True, x_log_scale=True, x_label='energy values', y_label='counts', plot_path=plot_path, name='energy_hist' + file_ext)
    if len(self.stats_dict['sample_num']): plot_histogram(self.stats_dict['sample_num'], bins=50, y_log_scale=True, x_label='sample numbers', y_label='counts', plot_path=plot_path, name='num_sample_hist' + file_ext)
    if len(self.stats_dict['damaged_score']): plot_histogram(self.stats_dict['damaged_score'], bins=50, y_log_scale=True, x_label='damaged score', y_label='counts', plot_path=plot_path, name='damaged_score_hist' + file_ext)

    # damaged file score 2d plot
    if len(self.stats_dict['damaged_score']): plot_damaged_file_score(self.stats_dict['damaged_score'], plot_path=self.plot_paths['stats'], name='damaged_score_hist' + file_ext)

    # print info
    print("too short files num: {}".format(len(self.info_file_dict['short']))), print("too weak files num: {}".format(len(self.info_file_dict['weak']))), print("damaged files num: {}".format(len(self.info_file_dict['damaged'])))

    # save damaged files
    for wav, score in self.info_file_dict['damaged']: copyfile(wav, self.plot_paths['damaged_files'] + wav.split('/')[-1])

    # save info files
    for k, info_list in self.info_file_dict.items():

      # skip if empty
      if not len(info_list): continue

      # file name
      file_name = plot_path + 'info_file_list_{}{}.txt'.format(k, file_ext)

      # write file
      with open(file_name, 'w') as f: [print(i, file=f) for i in info_list]


  def file_naming_extraction(self, audio_file, file_ext='.wav'):
    """
    extracts the file name, index and label of a file
    convention to filename: e.g. label123.wav
    """

    # extract filename
    file_name = re.findall(r'[\w+ 0-9 -]+' + re.escape(file_ext), audio_file)[0]

    # remove .wav and split
    file_name_split = re.sub(re.escape(file_ext), '', file_name).split('--')

    # extract my name
    file_name_no_ext = file_name_split[0]

    # hash name
    file_name_hash = file_name_split[1] if len(file_name_split) == 2 else 'no_hash'

    # extract label from filename
    file_label = re.sub(r'[0-9]+', '', file_name_no_ext)

    # extract file index from filename
    file_index = re.sub(r'[a-z A-Z]', '', file_name_no_ext)

    return file_name, file_label, file_index, file_name_hash


  def get_labeled_wavs_from_file(self, label, file):
    """
    get labeled wavs from file
    """

    # init label wavs
    label_wavs = []

    # read file
    with open(file, "r") as f:

      # read each line
      label_wavs = [self.dataset_path + line.strip() for line in f if line.split('/')[0] == label]

    return label_wavs



class SpeechCommandsDataset(AudioDataset):
  """
  Speech Commands Dataset extraction and set creation
  """

  def __init__(self, cfg_dataset, feature_params):

    # parent init
    super().__init__(cfg_dataset, feature_params)

    # create plot plaths if not already exists
    create_folder(list(self.plot_paths.values()))

    # recreate
    if self.cfg_dataset['recreate'] or not check_folders_existance(self.wav_folder_dict.values(), empty_check=True):

      # delete old data
      if self.cfg_dataset['clean_files']: delete_files_in_path(self.wav_folder_dict.values(), file_ext=self.cfg_dataset['file_ext'])

      # create folder wav folders
      create_folder(self.wav_folder_dict.values())

      # create sets (specific to dataset)
      self.create_sets()

    # get audio files from sets
    self.get_audiofiles()
    self.get_annotation_files()


  def create_sets(self):
    """
    copy wav files from dataset path to wav folders with splitting
    """

    # create set info
    print("\n--create wav sets for ", self.__class__.__name__)

    # noise
    self.create_noise_sets()

    # get all class directories except the ones starting with _
    class_dirs = glob(self.dataset_path + '[!_]*/')

    # run through all class directories
    for class_dir in class_dirs:

      # extract label
      label = class_dir.split('/')[-2]

      # get all .wav files
      wavs = glob(class_dir + '*' + self.cfg_dataset['file_ext'])

      # shuffle
      if self.cfg_dataset['shuffle_wavs']: np.random.seed(9999), np.random.shuffle(wavs)

      # validation wavs from file
      validation_wavs = self.get_labeled_wavs_from_file(label, file=self.dataset_path + self.cfg_dataset['validation_file_list'])

      # testing wavs from file
      testing_wavs = self.get_labeled_wavs_from_file(label, file=self.dataset_path + self.cfg_dataset['testing_file_list'])

      # training wavs
      training_wavs = [wav for wav in wavs if wav not in validation_wavs and wav not in testing_wavs]

      # print some info
      print("label: [{}]\tn_split: [{}]\ttotal:[{}]".format(label, {'train': len(training_wavs), 'test': len(testing_wavs), 'validation': len(validation_wavs)}, np.sum([len(training_wavs), len(validation_wavs), len(testing_wavs)])))

      # copy set wavs
      for set_name, set_wavs in {'train': training_wavs, 'test': testing_wavs, 'validation': validation_wavs}.items(): [copyfile(wav, self.wav_folder_dict[set_name] + label + '{:0>4}'.format(i) + '--' + wav.split('/')[-1].split('.')[0] + self.cfg_dataset['file_ext']) for i, wav in enumerate(set_wavs) if not os.path.isfile(self.wav_folder_dict[set_name] + label + '{:0>4}'.format(i) + '--' + wav.split('/')[-1].split('.')[0] + self.cfg_dataset['file_ext'])]


  def create_noise_sets(self):
    """
    noise sets
    """

    # create noise wavs
    noise_files = glob(self.dataset_path + self.cfg_dataset['noise_data_folder'] + '*' + self.cfg_dataset['file_ext'])

    # all noise wavs
    x_noise = np.empty(shape=(0, self.feature_params['fs']), dtype=np.float32)

    # go through all noise files
    for noise_file in noise_files:

      # read audio from file
      x, fs = librosa.load(noise_file, sr=self.feature_params['fs'])

      # windowing the noise
      x_win = view_as_windows(x, self.feature_params['fs'], step=int(self.feature_params['fs'] * self.cfg_dataset['noise_shift_s']))

      # stack noise to wavs
      x_noise = np.vstack((x_noise, x_win))
    
    # shuffle
    np.random.seed(6666)
    x_noise = np.take(x_noise, np.random.permutation(x_noise.shape[0]), axis=0)

    # number files for splits
    n_splits = {k: int(x_noise.shape[0] * v) for k, v in self.cfg_dataset['split_percs'].items()}

    # determine start and end positions
    n_split_end_pos = {k: v for k, v in zip(n_splits.keys(), np.cumsum(list(n_splits.values())))}
    n_split_start_pos = {k: w - v for (k, v), w in zip(n_splits.items(), n_split_end_pos.values())}

    # do splits
    x_noise_split = {k: x_noise[n_split_start_pos[k]:n_split_end_pos[k]] for k in n_splits.keys()}

    # print some info
    print("label: [{}]\tn_split: [{}]\ttotal:[{}]".format(self.cfg_dataset['noise_label'], n_splits, np.sum(list(n_splits.values()))))

    # for all sets
    for set_name, x_set in x_noise_split.items():

      # for all examples
      for i, x in enumerate(x_set):

        # determine wav name
        wav_file_name = self.wav_folder_dict[set_name] + self.cfg_dataset['noise_label'] + '{:0>4}'.format(i) + self.cfg_dataset['file_ext']

        # copy file
        if not os.path.isfile(wav_file_name): soundfile.write(wav_file_name, x, self.feature_params['fs'], subtype=None, endian=None, format=None, closefd=True)


  def extract_features(self):
    """
    extract mfcc features and save them
    """

    print("\n--feature extraction of [{}]:".format('mfcc' if self.feature_params['use_mfcc_features'] else 'raw'))

    # stop if already exists
    if check_files_existance(self.feature_file_dict.values()) and not self.cfg_dataset['recreate']:
      print("*** feature files already exists -> no extraction")
      return
    
    # create folder structure
    create_folder(self.feature_folder_dict.values())

    for ((set_name, wav_dict), (_, annos)) in zip(self.sel_label_file_dict.items(), self.set_annotation_file_dict.items()):

      print("\nextract set: [{}] with labels: [{}]".format(set_name, list(wav_dict.keys())))

      # examples with splits
      n_examples = int(self.cfg_dataset['n_examples'] * self.cfg_dataset['split_percs'][set_name])

      # extract data
      x, y, t, z = self.extract_wavs_to_features(wav_dict=wav_dict, annos=annos, n_examples=n_examples, set_name=set_name)

      # print label stats
      self.label_stats(y)

      # save data files
      np.savez(self.feature_file_dict[set_name], x=x, y=y, t=t, z=z, cfg_dataset=self.cfg_dataset, feature_params=self.feature_params), print("--saved data to: ", self.feature_file_dict[set_name])


  def extract_wavs_to_features(self, wav_dict, annos, n_examples, set_name=None):
    """
    extract given wavs to features, x: values, y: labels, t: target (for wavenets), z:index
    """

    # init collections
    x_data = np.empty(shape=(0, self.channel_size, self.feature_size, self.frame_size), dtype=np.float64) if self.feature_params['use_mfcc_features'] else np.empty(shape=(0, self.channel_size, self.raw_frame_size), dtype=np.float64)
    t_data = None if self.feature_params['use_mfcc_features'] else np.empty(shape=(0, self.raw_frame_size), dtype=np.int64)
    y_data, z_data =  [], []

    # extract class wavs
    for (label, wavs), class_annos in zip(wav_dict.items(), annos):

      print("extract label: ", label)

      # class annotation file names extraction
      class_annos_file_names = [l + i for f, l, i, h in [self.file_naming_extraction(a, file_ext='.TextGrid') for a in class_annos]]

      # class examples counter
      num_class_examples = 0

      # run through each example in class wavs
      for wav in wavs:
        
        # extract file namings
        file_name, file_label, file_index, file_name_hash = self.file_naming_extraction(wav, file_ext=self.cfg_dataset['file_ext'])

        # get annotation if available
        anno = class_annos[class_annos_file_names.index(file_label + file_index)] if file_label + file_index in class_annos_file_names else None

        # load and pre-process audio
        x, wav_is_useless = self.wav_pre_processing(wav)
        if wav_is_useless: continue

        # print some info
        if self.cfg_dataset['verbose']: print("wav: [{}] with label: [{}], samples=[{}], time=[{}]s".format(wav, label, len(x), len(x) / self.feature_params['fs']))

        # extract features
        x_feature, bon_pos = self.feature_extractor.extract_audio_features(x, reduce_to_best_onset=False, rand_best_onset=self.cfg_dataset['rand_best_onset'], rand_delta_percs=self.cfg_dataset['rand_delta_percs'])

        # quantize data (only for raw features)
        t = self.feature_extractor.quantize(np.squeeze(x_feature)[bon_pos:bon_pos+self.raw_frame_size]) if not self.feature_params['use_mfcc_features'] else None

        # plot features
        if self.cfg_dataset['enable_plot']: plot_mfcc_profile(x, self.feature_params['fs'], self.feature_extractor.N, self.feature_extractor.hop, x_feature, anno_file=anno, onsets=None, bon_pos=bon_pos, mient=None, minreg=None, frame_size=self.frame_size, plot_path=self.plot_paths['mfcc'], name=file_label + str(file_index) + '_' + set_name, close_plot=True) if self.feature_params['use_mfcc_features'] else plot_waveform(x, self.feature_params['fs'],  bon_samples=[bon_pos, bon_pos+self.raw_frame_size], title=file_label + file_index, plot_path=self.plot_paths['waveform'], name=file_label + file_index, show_plot=False, close_plot=True)

        # damaged file check
        if self.cfg_dataset['filter_damaged_files'] and self.feature_params['use_mfcc_features']:

          # handle damaged files
          if self.detect_damaged_file(x_feature, wav): continue


        # add data to container
        x_data = np.vstack((x_data, x_feature[np.newaxis, :, :, bon_pos:bon_pos+self.frame_size])) if self.feature_params['use_mfcc_features'] else np.vstack((x_data, x_feature[np.newaxis, :, bon_pos:bon_pos+self.raw_frame_size]))

        # add target to container
        t_data = np.vstack((t_data, t)) if not self.feature_params['use_mfcc_features'] else None

        # add label and index to container
        y_data.append(label), z_data.append(file_name)


        # update number of examples per class
        num_class_examples += 1

        # stop if desired examples are reached
        if num_class_examples >= n_examples: break

    return x_data, y_data, t_data, z_data


  def detect_damaged_file(self, mfcc, wav):
    """
    detect if file is damaged
    """

    # energy calc
    #e = np.einsum('ij,ji->j', mfcc, mfcc.T)
    #e = e / np.max(e)

    # calculate damaged score of energy deltas
    if mfcc.shape[1] == 39: z_est, z_lim = np.sum(np.abs(mfcc[0, 37:39, :])), 60
    #if mfcc.shape[0] == 39: z_est = np.sum(mfcc[37:39, :] @ mfcc[37:39, :].T)
    #else: z_est = np.sum(np.abs(np.diff(mfcc[-1, :])))
    #else: z_est = np.diff(mfcc[-1, :]) @ np.diff(mfcc[-1, :]).T
    #else: z_est = np.sum(mfcc[0, :])
    #else: z_est = np.diff(mfcc[0, :]) @ np.diff(mfcc[0, :]).T
    #else: z_est = np.abs(np.diff(mfcc[0, :])) @ mfcc[0, :-1].T
    #else: z_est = np.sum(np.diff(mfcc, axis=1) @ np.diff(mfcc, axis=1).T)
    #else: z_est = np.sum(e)
    #else: z_est = np.diff(e) @ np.diff(e).T
    else: z_est, z_lim = mfcc[0, 0, :-1] @ np.abs(np.diff(mfcc[0, 0, :])).T, 3.5

    # add score to list
    self.stats_dict['damaged_score'].append(z_est)

    # damaged file
    is_damaged = z_est > z_lim

    # add to damaged file list
    if is_damaged: self.info_file_dict['damaged'].append((wav, z_est))

    # return score and damaged indicator
    return is_damaged



class MyRecordingsDataset(SpeechCommandsDataset):
  """
  Speech Commands Dataset extraction and set creation
  """

  def create_sets(self):
    """
    cut and copy recorded wavs
    """

    #print("wav: ", self.wav_folders)
    print("wav folders: ", self.wav_folder_dict)

    # get all .wav files
    raw_wavs = glob(self.dataset_path + '*' + self.cfg_dataset['file_ext'])

    # get all wav files and save them
    for i, wav in enumerate(raw_wavs):

      print("wav: ", wav)

      # filename extraction
      file_name, label, file_index, file_name_hash = self.file_naming_extraction(wav, file_ext=self.cfg_dataset['file_ext'])

      # read audio from file
      x, _ = librosa.load(wav, sr=self.feature_params['fs'])

      # calc onsets
      onsets = calc_onsets(x, self.feature_params['fs'], N=self.N, hop=self.hop, adapt_frames=5, adapt_alpha=0.09, adapt_beta=0.8)
      onsets = self.clean_onsets(onsets)

      # cut examples to one second
      x_cut = self.cut_signal(x, onsets, time=1, alpha=0.4)

      # plot onsets
      plot_onsets(x, self.feature_params['fs'], self.N, self.hop, onsets, title=label, plot_path=self.plot_paths['onsets'], name='onsets_{}'.format(label))

      for j, xj in enumerate(x_cut):

        # plot
        plot_waveform(xj, self.feature_params['fs'], title='{}-{}'.format(label, j), plot_path=self.plot_paths['waveform'], name='example_{}-{}'.format(label, j), close_plot=True)

        # save file
        soundfile.write('{}{}{}.wav'.format(self.wav_folder_dict['my'], label, j), xj, self.feature_params['fs'], subtype=None, endian=None, format=None, closefd=True)


  def clean_onsets(self, onsets):
    """
    clean onsets, so that only one per examples exists
    """

    # analytical frame size
    analytic_frame_size = self.frame_size * 3

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
        onsets[i:i+analytic_frame_size] = onset_filter[:len(onsets[i:i+analytic_frame_size])]
        clean_onsets[i] = 1

    return clean_onsets


  def cut_signal(self, x, onsets, time=1, alpha=0.5):
    """
    cut signal at onsets to a specific time interval
    """

    # init cut signal
    x_cut = np.zeros((int(np.sum(onsets)), int(self.feature_params['fs'] * time)))

    # amount of samples 
    num_samples = self.feature_params['fs'] / time

    # pre and post samples
    pre, post = int(alpha * num_samples), int((1 - alpha) * num_samples)

    # onset index
    oi = 0

    # check all onsets
    for i, onset in enumerate(onsets):

      # cut accordingly
      if onset:
        x_cut[oi, :] = x[i*self.hop-pre:i*self.hop+post]
        oi += 1

    return x_cut



if __name__ == '__main__':
  """
  main function of audio dataset
  """

  import yaml
  from batch_archive import SpeechCommandsBatchArchive
  from plots import plot_mfcc_only, plot_wav_grid

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # audioset init
  audio_set1 = SpeechCommandsDataset(cfg['datasets']['speech_commands'], feature_params=cfg['feature_params'])
  audio_set2 = MyRecordingsDataset(cfg['datasets']['my_recordings'], feature_params=cfg['feature_params'])

  # extract and save features
  audio_set1.extract_features()
  audio_set2.extract_features()

  # analyze audio set
  audio_set1.analyze_dataset_extraction(calculate_overall_stats=False)
  audio_set2.analyze_dataset_extraction(calculate_overall_stats=False)


  print("\n--check dataset with batch archive:")

  # batches
  batch_archive = SpeechCommandsBatchArchive(feature_file_dict={**audio_set1.feature_file_dict, **audio_set2.feature_file_dict}, batch_size_dict={'train': 32, 'test': 5, 'validation': 5, 'my': 1}, shuffle=False)

  # create batches
  batch_archive.create_batches()

  # show info
  batch_archive.print_batch_infos()