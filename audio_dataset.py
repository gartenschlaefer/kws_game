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
from common import check_folders_existance, create_folder, delete_files_in_path


class AudioDataset():
  """
  audio dataset interface with some useful functions
  """

  def __init__(self, dataset_cfg, feature_params, root_path='./'):

    # params
    self.dataset_cfg = dataset_cfg
    self.feature_params = feature_params
    self.root_path = root_path

    # extracted vars
    self.feature_size = (self.feature_params['n_ceps_coeff'] + 1) * (1 + 2 * self.feature_params['compute_deltas'])

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
    self.param_path = 'v{}_c-{}_n-{}_f-{}x{}/'.format(self.dataset_cfg['version_nr'], len(self.labels), self.dataset_cfg['n_examples'], self.feature_size, self.feature_params['frame_size'])

    # folders
    self.wav_folders = [self.root_path + p + self.dataset_cfg['wav_folder'] for p in list(self.dataset_cfg['data_paths'].values())]
    self.annotation_folders = [self.root_path + p + self.dataset_cfg['annotation_folder'] for p in list(self.dataset_cfg['data_paths'].values())]

    # feature folders
    self.feature_folders = [self.root_path + p + self.param_path for p in list(self.dataset_cfg['data_paths'].values())]

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


  def file_naming_extraction(self, audio_file, file_ext='.wav'):
    """
    extracts the file name, index and label of a file
    convention to filename: e.g. label123.wav
    """

    # extract filename
    file_name = re.findall(r'[\w+ 0-9]+' + re.escape(file_ext), audio_file)[0]

    # extract file index from filename
    file_index = re.sub(r'[a-z A-Z]|(' + re.escape(file_ext) + r')', '', file_name)

    # extract label from filename
    label = re.sub(r'([0-9]+' + re.escape(file_ext) + r')', '', file_name)

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



class SpeechCommandsDataset(AudioDataset):
  """
  Speech Commands Dataset extraction and set creation
  """

  def __init__(self, dataset_cfg, feature_params, verbose=False):

    # parent init
    super().__init__(dataset_cfg, feature_params)

    # arguments
    self.verbose = verbose

    # feature extractor
    self.feature_extractor = FeatureExtractor(feature_params=self.feature_params)

    # short vars
    self.N = self.feature_extractor.N
    self.hop = self.feature_extractor.hop

    # minimum amount of samples per example
    self.min_samples = 16000

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

    # create folder structure
    create_folder(self.feature_folders)

    for i, (set_name, wavs, annos) in enumerate(zip(self.set_names, self.set_audio_files, self.set_annotation_files)):

      print("{}) extract set: {} with label num: {}".format(i, set_name, len(wavs)))

      # examples with splits
      n_examples = int(self.dataset_cfg['n_examples']*self.dataset_cfg['split_percs'][i])

      # extract data
      x, y, index = self.extract_mfcc_data(wavs=wavs, annos=annos, n_examples=n_examples, set_name=set_name)

      # print label stats
      self.label_stats(y)

      # save mfcc data file
      np.savez(self.feature_files[i], x=x, y=y, index=index, params=cfg['feature_params'])
      print("--save data to: ", self.feature_files[i])


  def extract_mfcc_data(self, wavs, annos, n_examples, set_name=None):
    """
    extract mfcc data from wav-files
    wavs must be in a 2D-array [[wavs_class1], [wavs_class2]] so that n_examples will work properly
    """

    # mfcc_data: [n x m x l], labels and index
    mfcc_data, label_data, index_data = np.empty(shape=(0, self.feature_size, self.feature_params['frame_size']), dtype=np.float64), [], []

    # some lists
    z_score_list, broken_file_list,  = np.array([]), []

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
        x = self.wav_pre_processing(wav)

        # print some info
        if self.verbose:
          print("wav: [{}] with label: [{}], samples=[{}], time=[{}]s".format(wav, label, len(x), len(x)/self.feature_params['fs']))

        # extract feature vectors [m x l]
        mfcc, bon_pos = self.feature_extractor.extract_mfcc39(x, reduce_to_best_onset=False)

        # damaged file things
        z_score, z_damaged = self.detect_damaged_file(mfcc)
        z_score_list = np.append(z_score_list, z_score)

        # plot mfcc features
        plot_mfcc_profile(x, self.feature_params['fs'], self.feature_extractor.N, self.feature_extractor.hop, mfcc, anno_file=anno, onsets=None, bon_pos=bon_pos, mient=None, minreg=None, frame_size=self.feature_params['frame_size'], plot_path=self.plot_paths['mfcc'], name=label + str(file_index) + '_' + set_name, enable_plot=self.dataset_cfg['enable_plot'])

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
    plot_damaged_file_score(z_score_list, plot_path=self.plot_paths['z_score'], name='z_score_n-{}_{}'.format(n_examples, set_name), enable_plot=self.dataset_cfg['enable_plot'])
    
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
  from plots import plot_mfcc_only

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # audioset init
  audio_set1 = SpeechCommandsDataset(cfg['datasets']['speech_commands'], feature_params=cfg['feature_params'], verbose=False)
  audio_set2 = MyRecordingsDataset(cfg['datasets']['my_recordings'], feature_params=cfg['feature_params'], verbose=False)

  # extract and save features
  audio_set1.extract_features()
  audio_set2.extract_features()

  # select feature files
  all_feature_files = audio_set1.feature_files + audio_set2.feature_files if len(audio_set1.labels) == len(audio_set2.labels) else audio_set1.feature_files

  # batches
  batch_archive = SpeechCommandsBatchArchive(feature_files=all_feature_files, batch_size=32, batch_size_eval=4, to_torch=False)

  print("archive: ", batch_archive.x_train.shape)
  print("archive: ", batch_archive.num_examples_per_class)
  plot_mfcc_only(batch_archive.x_train[0, 0], name=batch_archive.z_train[0, 0], show_plot=True)






