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
from plots import plot_mfcc_profile, plot_damaged_file_score, plot_onsets, plot_waveform, plot_histogram
from common import check_folders_existance, create_folder, delete_files_in_path


class AudioDataset():
  """
  audio dataset interface with some useful functions
  """

  def __init__(self, dataset_cfg, feature_params, collect_wavs=False, verbose=False, root_path='./'):

    # params
    self.dataset_cfg = dataset_cfg
    self.feature_params = feature_params
    self.collect_wavs = collect_wavs
    self.verbose = verbose
    self.root_path = root_path

    # channel size
    self.channel_size = 1 if not self.feature_params['use_channels'] else int(self.feature_params['use_cepstral_features']) + int(self.feature_params['use_delta_features']) +  int(self.feature_params['use_double_delta_features'])

    # feature size
    self.feature_size = (self.feature_params['n_ceps_coeff'] + int(self.feature_params['use_energy_features'])) * int(self.feature_params['use_cepstral_features']) + (self.feature_params['n_ceps_coeff'] + int(self.feature_params['use_energy_features'])) * int(self.feature_params['use_delta_features']) + (self.feature_params['n_ceps_coeff'] + int(self.feature_params['use_energy_features'])) * int(self.feature_params['use_double_delta_features']) if not self.feature_params['use_channels'] else (self.feature_params['n_ceps_coeff'] + int(self.feature_params['use_energy_features']))
    
    # variables
    self.labels = self.dataset_cfg['sel_labels']
    self.set_names = []
    self.set_audio_files = []
    self.set_annotation_files = []

    # dataset path
    self.dataset_path = self.root_path + self.dataset_cfg['dataset_path']

    # plot paths
    self.plot_paths = dict((k, self.root_path + v) for k, v in self.dataset_cfg['plot_paths'].items())

    # parameter path
    #self.param_path = 'v{}_c-{}_n-{}_f-{}x{}_n{}d{}e{}_nl-{}/'.format(self.dataset_cfg['version_nr'], len(self.labels), self.dataset_cfg['n_examples'], self.feature_size, self.feature_params['frame_size'], int(self.feature_params['norm_features']), int(self.feature_params['compute_deltas']), int(self.feature_params['compute_energy_features']), int(self.dataset_cfg['add_noise']))
    self.param_path = 'v{}_c-{}_n-{}_f-{}x{}x{}_norm{}_c{}d{}d{}e{}_nl{}/'.format(self.dataset_cfg['version_nr'], len(self.labels), self.dataset_cfg['n_examples'], self.channel_size, self.feature_size, self.feature_params['frame_size'], int(self.feature_params['norm_features']), int(self.feature_params['use_cepstral_features']), int(self.feature_params['use_delta_features']), int(self.feature_params['use_double_delta_features']), int(self.feature_params['use_energy_features']), int(self.dataset_cfg['add_noise']))

    # folders
    self.wav_folders = [self.root_path + p + self.dataset_cfg['wav_folder'] for p in list(self.dataset_cfg['data_paths'].values())]
    self.annotation_folders = [self.root_path + p + self.dataset_cfg['annotation_folder'] for p in list(self.dataset_cfg['data_paths'].values())]

    # feature folders
    self.feature_folders = [self.root_path + p + self.param_path for p in list(self.dataset_cfg['data_paths'].values())]

    # feature files
    self.feature_files = [p + '{}.npz'.format(self.dataset_cfg['feature_file_name']) for p in self.feature_folders]

    # collected wavs
    self.pre_wavs = []
    self.short_file_list, self.file_num_sample_list, self.damaged_file_list, self.damaged_score_list,  = [], [], [], []
    self.file_energy_list, self.weak_file_list, self.strong_file_list = [], [], []

    # info files
    self.info_file_damaged = self.plot_paths['z_score'] + 'info_damaged_file_list_n-{}.txt'.format(self.dataset_cfg['n_examples'])
    self.info_file_short = self.plot_paths['z_score'] + 'info_short_file_list_n-{}.txt'.format(self.dataset_cfg['n_examples'])
    self.info_file_weak = self.plot_paths['z_score'] + 'info_weak_file_list_n-{}.txt'.format(self.dataset_cfg['n_examples'])
    self.info_file_strong = self.plot_paths['z_score'] + 'info_strong_file_list_n-{}.txt'.format(self.dataset_cfg['n_examples'])

    # ignore list for damaged files
    try:
      with open(self.plot_paths['damaged_files'] + 'damaged_file_list.txt') as f:
        self.ignore_file_list_re = '(' + ')|('.join([line.strip() for line in f]) +')'
    except: 
      self.ignore_file_list_re = []


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


  def analyze_damaged_files(self):
    """
    analyze damaged files
    """

    # histogram
    if len(self.file_energy_list): plot_histogram(self.file_energy_list, bins=np.logspace(np.log10(0.0001),np.log10(10000), 50), y_log_scale=True, x_log_scale=True, context='None', title='Energy', plot_path=self.plot_paths['z_score'], name='energy_hist_n-{}'.format(self.dataset_cfg['n_examples']))
    if len(self.file_num_sample_list): plot_histogram(self.file_num_sample_list, bins=20, y_log_scale=True, context='None', title='Num Samples', plot_path=self.plot_paths['z_score'], name='num_sample_hist_n-{}'.format(self.dataset_cfg['n_examples']))
    if len(self.damaged_score_list): plot_histogram(self.damaged_score_list, bins=50, y_log_scale=True, context='None', title='Damaged Score', plot_path=self.plot_paths['z_score'], name='z_score_hist_n-{}'.format(self.dataset_cfg['n_examples']))

    print("\n--Analyze damaged files of ", self.__class__.__name__)
    print("too short files num: {}".format(len(self.short_file_list)))
    print("too weak files num: {}".format(len(self.weak_file_list)))
    print("damaged files num: {}".format(len(self.damaged_file_list)))

    # all audio files speakers
    if self.__class__.__name__ == 'SpeechCommandsDataset':
      all_speakers_files = [re.sub(r'(\./)|(\w+/)|(\w+--)|(_nohash_[0-9]+.wav)', '', i) for i in list(np.concatenate(np.concatenate(np.array(self.set_audio_files, dtype='object'))))]
      all_speakers = np.unique(all_speakers_files)
      print("number of audio files: ", len(all_speakers_files)), print("speakers: ", all_speakers), print("number of speakers: ", len(all_speakers))

      # save damaged files
      for wav, score in self.damaged_file_list: copyfile(wav, self.plot_paths['damaged_files'] + wav.split('/')[-1])
    
      # prints to files
      with open(self.info_file_damaged, 'w') as f: [print(i, file=f) for i in self.damaged_file_list]
      with open(self.info_file_short, 'w') as f: [print(i, file=f) for i in self.short_file_list]
      with open(self.info_file_weak, 'w') as f: [print(i, file=f) for i in self.weak_file_list]
      with open(self.info_file_strong, 'w') as f: [print(i, file=f) for i in self.strong_file_list]

    # broken file info
    plot_damaged_file_score(self.damaged_score_list, plot_path=self.plot_paths['z_score'], name='z_score_n-{}'.format(self.dataset_cfg['n_examples']), enable_plot=True)


  def file_naming_extraction(self, audio_file, file_ext='.wav'):
    """
    extracts the file name, index and label of a file
    convention to filename: e.g. label123.wav
    """

    # extract filename
    file_name = re.findall(r'[\w+ 0-9 -]+' + re.escape(file_ext), audio_file)[0]

    # extract my name
    file_name_no_ext = re.sub(re.escape(file_ext), '', file_name.split('--')[0])

    # extract file index from filename
    file_index = re.sub(r'[a-z A-Z]', '', file_name_no_ext)

    # extract label from filename
    label = re.sub(r'[0-9]+', '', file_name_no_ext)

    return file_name, file_index, label


  def get_audiofiles(self):
    """
    get audiofiles from datapaths
    return [[data_path1], [data_path2], ...]
    """

    # for all data paths (train, test, eval)
    for dpi, wav_folder in enumerate(self.wav_folders):

      # info
      print("\nset wav folder: {}".format(wav_folder))

      # determine set name
      self.set_names.append(re.sub(r'/', '', re.findall(r'[\w+ 0-9]+/', wav_folder)[-2]))

      # init set files
      set_files = []

      # get all wavs from selected labels
      for l in self.labels:

        # regex
        file_name_re = '*' + l + '[0-9]*' + self.dataset_cfg['file_ext']

        # get wavs
        label_files = glob(wav_folder + file_name_re)
        set_files.append(label_files)

        # check length of label files
        print("overall stat of label: [{}]\tnum: [{}]".format(l, len(label_files)))

        # check label num
        if len(label_files) < int(self.dataset_cfg['n_examples'] * self.dataset_cfg['split_percs'][dpi]):
          print("***[audio set] labels are less than n_examples, recreate dataset and check files")
          import sys
          sys.exit()

      # update set audio files
      self.set_audio_files.append(set_files)


  def get_annotation_files(self):
    """
    get annotation files
    """

    # for all data paths (train, test, eval)
    for dpi, annotation_folder in enumerate(self.annotation_folders):

      # info
      print("\nset annotation folder: {}".format(annotation_folder))

      # init set files
      set_files = []

      # get all wavs from selected labels
      for l in self.labels:

        # regex
        file_name_re = '*' + l + '[0-9]*' + '.TextGrid'

        # get wavs
        label_files = glob(annotation_folder + file_name_re)
        set_files.append(label_files)

        # check length of label files
        print("overall stat of anno: [{}]\tnum: [{}]".format(l, len(label_files)))

      # update set audio files
      self.set_annotation_files.append(set_files)


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


  def add_noise_to_dataset(self, x, y, z, n_examples):
    """
    add noise data
    """

    # create noise
    n = np.random.rand(n_examples, x.shape[1], x.shape[2], x.shape[3])

    # update
    x = np.vstack((x, n))
    y = y + [self.dataset_cfg['noise_label'] for i in range(n_examples)]
    z = z + [self.dataset_cfg['noise_label'] + str(i) for i in range(n_examples)]

    return x, y, z


  def wav_pre_processing(self, wav):
    """
    audio pre processing, check if file is okay and clean up a bit, no normalization
    """

    # check ignore file list
    if len(self.ignore_file_list_re):
      ignore = re.findall(self.ignore_file_list_re, wav)
      if len(ignore):
        print("ignore: ", ignore)
        print("wav: ", wav)
        return 0, True

    # read audio from file
    x_raw, fs = librosa.load(wav, sr=self.feature_params['fs'])

    # energy of raw audio
    e = x_raw @ x_raw.T

    # save amount of samples
    self.file_num_sample_list.append(len(x_raw))
    self.file_energy_list.append(e)

    # too short flag
    is_too_short = len(x_raw) < self.dataset_cfg['sample_num_mininmal']
    is_too_weak = e < 0.01
    is_too_strong = e > 1000.0

    # too short
    if is_too_short: self.short_file_list.append((wav, len(x_raw)))
    if is_too_weak: self.weak_file_list.append((wav, e))
    if is_too_strong: self.strong_file_list.append((wav, e))

    # check sample lengths
    if len(x_raw) < self.dataset_cfg['sample_num_normal']:

      # print warning
      if self.verbose: print("lengths is less than 1s, append with zeros for:")

      # append with zeros
      x_raw = np.append(x_raw, np.zeros(self.dataset_cfg['sample_num_normal'] - len(x_raw)))

    return x_raw, is_too_weak



class SpeechCommandsDataset(AudioDataset):
  """
  Speech Commands Dataset extraction and set creation
  """

  def __init__(self, dataset_cfg, feature_params, collect_wavs=False, verbose=False):

    # parent init
    super().__init__(dataset_cfg, feature_params, collect_wavs=collect_wavs, verbose=verbose)

    # feature extractor
    self.feature_extractor = FeatureExtractor(feature_params=self.feature_params)

    # short vars
    self.N = self.feature_extractor.N
    self.hop = self.feature_extractor.hop

    # create plot plaths if not already exists
    create_folder(list(self.plot_paths.values()))

    # recreate
    if self.dataset_cfg['recreate'] or not check_folders_existance(self.wav_folders, empty_check=True):

      # delete old data
      delete_files_in_path(self.wav_folders, file_ext=self.dataset_cfg['file_ext'])

      # create folder wav folders
      create_folder(self.wav_folders)

      # create sets (specific to dataset)
      self.create_sets()

    # get audio files from sets
    self.get_audiofiles()
    self.get_annotation_files()


  def create_sets(self):
    """
    copy wav files from dataset path to wav folders with splitting
    """

    # get all class directories except the ones starting with _
    class_dirs = glob(self.dataset_path + '[!_]*/')

    # run through all class directories
    for class_dir in class_dirs:

      # extract label
      label = class_dir.split('/')[-2]

      # get all .wav files
      wavs = glob(class_dir + '*' + self.dataset_cfg['file_ext'])

      # calculate split numbers in train, test, eval and split position
      n_split = (len(wavs) * np.array(self.dataset_cfg['split_percs'])).astype(int)
      n_split_pos = np.cumsum(n_split)

      # print some info
      print("label: [{}]\tn_split: [{}]\ttotal:[{}]".format(label, n_split, np.sum(n_split)))

      # actual path
      p = 0

      # shuffle
      if self.dataset_cfg['shuffle_wavs']: np.random.shuffle(wavs)

      # run through each path
      for i, wav in enumerate(wavs):

        # split in new path
        if i >= n_split_pos[p]: p += 1
        # stop if out of range (happens at rounding errors)
        if p >= len(self.wav_folders): break

        # wav name
        wav_name = wav.split('/')[-1].split('.')[0]

        # copy files to folder
        copyfile(wav, self.wav_folders[p] + label + str(i) + '--' + wav_name + self.dataset_cfg['file_ext'])


  def extract_features(self):
    """
    extract mfcc features and save them
    """

    print("\n--feature extraction:")

    # create folder structure
    create_folder(self.feature_folders)

    for i, (set_name, wavs, annos) in enumerate(zip(self.set_names, self.set_audio_files, self.set_annotation_files)):

      print("{}) extract set: {} with label num: {}".format(i, set_name, len(wavs)))

      # examples with splits
      n_examples = int(self.dataset_cfg['n_examples'] * self.dataset_cfg['split_percs'][i])

      # extract data
      x, y, index = self.extract_mfcc_data(wavs=wavs, annos=annos, n_examples=n_examples, set_name=set_name)

      # add noise if requested
      if self.dataset_cfg['add_noise']: x, y, index = self.add_noise_to_dataset(x, y, index, n_examples)

      # print label stats
      self.label_stats(y)

      # save mfcc data file
      np.savez(self.feature_files[i], x=x, y=y, index=index, params=self.feature_params)
      print("--save data to: ", self.feature_files[i])


  def extract_mfcc_data(self, wavs, annos, n_examples, set_name=None):
    """
    extract mfcc data from wav-files
    wavs must be in a 2D-array [[wavs_class1], [wavs_class2]] so that n_examples will work properly
    """

    # mfcc_data: [n x m x l], labels and index
    mfcc_data, label_data, index_data = np.empty(shape=(0, self.channel_size, self.feature_size, self.feature_params['frame_size']), dtype=np.float64), [], []

    # extract class wavs
    for class_wavs, class_annos in zip(wavs, annos):

      # class annotation file names extraction
      class_annos_file_names = [l + i for f, i, l in [self.file_naming_extraction(a, file_ext='.TextGrid') for a in class_annos]]

      # number of class examples
      num_class_examples = 0

      # run through each example in class wavs
      for wav in class_wavs:
        
        # extract file namings
        file_name, file_index, label = self.file_naming_extraction(wav, file_ext=self.dataset_cfg['file_ext'])

        # get annotation if available
        anno = None
        if label + file_index in class_annos_file_names: anno = class_annos[class_annos_file_names.index(label + file_index)]

        # load and pre-process audio
        x, wav_is_useless = self.wav_pre_processing(wav)
        if wav_is_useless: continue

        # print some info
        if self.verbose: print("wav: [{}] with label: [{}], samples=[{}], time=[{}]s".format(wav, label, len(x), len(x) / self.feature_params['fs']))

        # extract feature vectors [m x l]
        mfcc, bon_pos = self.feature_extractor.extract_mfcc(x, reduce_to_best_onset=False)

        # collect wavs
        if self.collect_wavs: self.pre_wavs.append((librosa.util.normalize(x), label + str(file_index) + '_' + set_name, bon_pos))

        # plot mfcc features
        plot_mfcc_profile(x, self.feature_params['fs'], self.feature_extractor.N, self.feature_extractor.hop, mfcc, anno_file=anno, onsets=None, bon_pos=bon_pos, mient=None, minreg=None, frame_size=self.feature_params['frame_size'], plot_path=self.plot_paths['mfcc'], name=label + str(file_index) + '_' + set_name, enable_plot=self.dataset_cfg['enable_plot'])

        # damaged file check
        if self.dataset_cfg['filter_damaged_files']:

          # handle damaged files
          if self.detect_damaged_file(mfcc, wav): continue

        # add to mfcc_data container
        mfcc_data = np.vstack((mfcc_data, mfcc[np.newaxis, :, :, bon_pos:bon_pos+self.feature_params['frame_size']]))
        label_data.append(label)
        index_data.append(label + file_index)

        # update number of examples per class
        num_class_examples += 1

        # stop if desired examples are reached
        if num_class_examples >= n_examples: break


    return mfcc_data, label_data, index_data


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
    self.damaged_score_list.append(z_est)

    # damaged file
    is_damaged = z_est > z_lim

    # add to damaged file list
    if is_damaged: self.damaged_file_list.append((wav, z_est))

    # return score and damaged indicator
    return is_damaged



class MyRecordingsDataset(SpeechCommandsDataset):
  """
  Speech Commands Dataset extraction and set creation
  """

  def __init__(self, dataset_cfg, feature_params, collect_wavs=False, verbose=False):

    # parent init
    super().__init__(dataset_cfg, feature_params, collect_wavs=collect_wavs, verbose=verbose)


  def create_sets(self):
    """
    cut and copy recorded wavs
    """

    print("wav: ", self.wav_folders)

    # get all .wav files
    raw_wavs = glob(self.dataset_path + '*' + self.dataset_cfg['file_ext'])

    # get all wav files and save them
    for i, wav in enumerate(raw_wavs):

      print("wav: ", wav)

      # filename extraction
      file_name, file_index, label = self.file_naming_extraction(wav, file_ext=self.dataset_cfg['file_ext'])

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
        plot_waveform(xj, self.feature_params['fs'], title='{}-{}'.format(label, j), plot_path=self.plot_paths['waveform'], name='example_{}-{}'.format(label, j))

        # save file
        soundfile.write('{}{}{}.wav'.format(self.wav_folders[0], label, j), xj, self.feature_params['fs'], subtype=None, endian=None, format=None, closefd=True)


  def clean_onsets(self, onsets):
    """
    clean onsets, so that only one per examples exists
    """

    # analytical frame size
    analytic_frame_size = self.feature_params['frame_size'] * 3

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
  from batch_archive import SpeechCommandsBatchArchive
  from plots import plot_mfcc_only, plot_wav_grid

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # audioset init
  audio_set1 = SpeechCommandsDataset(cfg['datasets']['speech_commands'], feature_params=cfg['feature_params'], verbose=False)
  audio_set2 = MyRecordingsDataset(cfg['datasets']['my_recordings'], feature_params=cfg['feature_params'], collect_wavs=True, verbose=False)

  # extract and save features
  audio_set1.extract_features()
  audio_set2.extract_features()

  # analyze audio set
  audio_set1.analyze_damaged_files()
  audio_set2.analyze_damaged_files()

  # select feature files
  all_feature_files = audio_set1.feature_files + audio_set2.feature_files if len(audio_set1.labels) == len(audio_set2.labels) else audio_set1.feature_files

  # batches
  batch_archive = SpeechCommandsBatchArchive(feature_files=all_feature_files, batch_size=32, batch_size_eval=4, to_torch=False)

  print("archive: ", batch_archive.x_train.shape)
  print("archive: ", batch_archive.num_examples_per_class)
  #plot_mfcc_only(batch_archive.x_train[0, 0], name=batch_archive.z_train[0, 0], show_plot=True)

  # plot wav grid
  plot_wav_grid(audio_set2.pre_wavs, feature_params=audio_set2.feature_params, grid_size=(5, 5), cmap=None, title='', plot_path=cfg['datasets']['my_recordings']['plot_paths']['examples_grid'], name='wav_grid_my', show_plot=True)






