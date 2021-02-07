"""
Path collector for files generated with input from config.yaml
"""

from common import create_folder


class PathCollector():
	"""
	includes all paths that must be generated from config.yaml
	"""

	def __init__(self, cfg, root_path=''):

		# config file
		self.cfg = cfg
		self.root_path = root_path

		# parameter path
		self.param_path_audio_dataset = 'v{}_c-{}_n-{}/'.format(self.cfg['audio_dataset']['version_nr'], len(self.cfg['audio_dataset']['sel_labels']), self.cfg['audio_dataset']['n_examples'])
		self.param_path_my_recordings = 'v{}_c-{}_n-{}/'.format(self.cfg['audio_dataset']['version_nr'], len(self.cfg['my_recordings']['sel_labels']), self.cfg['my_recordings']['n_examples']) 
		self.param_path_ml = 'bs-{}_it-{}_lr-{}/'.format(cfg['ml']['train_params']['batch_size'], cfg['ml']['train_params']['num_epochs'], str(cfg['ml']['train_params']['lr']).replace('.', 'p'))

		# mfcc files
		self.mfcc_data_files = [p + self.param_path_audio_dataset + '{}.npz'.format(self.cfg['audio_dataset']['mfcc_file_name']) for p in list(self.cfg['audio_dataset']['data_paths'].values())]
		self.mfcc_data_file_my = self.cfg['my_recordings']['out_path_root'] + self.param_path_my_recordings + '{}.npz'.format(self.cfg['my_recordings']['mfcc_file_name'])
		self.mfcc_data_files_all = self.mfcc_data_files + [self.mfcc_data_file_my]

		# wav folders in data paths for audio dataset
		self.wav_folders_audio_dataset = [p + self.cfg['audio_dataset']['wav_folder'] for p in list(self.cfg['audio_dataset']['data_paths'].values())]

		# model path
		self.model_path = self.cfg['ml']['paths']['model'] + self.cfg['ml']['nn_arch'] + '/' + self.param_path_audio_dataset + self.param_path_ml
		
		# model and params files
		self.model_file = self.model_path + self.cfg['ml']['model_file_name']
		self.model_pre_file = self.cfg['ml']['paths']['model_pre'] + '{}_c-{}.pth'.format(self.cfg['ml']['nn_arch'], len(self.cfg['audio_dataset']['sel_labels']))

		# for adversarial neural networks
		self.adv_g_model_file = self.model_path + self.cfg['ml']['adv_G_model_file_name']
		self.adv_d_model_file = self.model_path + self.cfg['ml']['adv_D_model_file_name']
		
		# params and metrics files
		self.params_file = self.model_path + self.cfg['ml']['params_file_name']
		self.metrics_file = self.model_path + self.cfg['ml']['metrics_file_name']

		# classifier model
		self.classifier_model = self.root_path + self.cfg['classifier']['model_path'] + self.cfg['classifier']['model_file_name']
		self.classifier_params = self.root_path + self.cfg['classifier']['model_path'] + self.cfg['classifier']['params_file_name']


	def create_audio_dataset_folders(self):
		"""
		create all necessary folders for audio dataset
		"""

		# mfcc paths for output
		mfcc_paths = [p + self.param_path_audio_dataset for p in list(self.cfg['audio_dataset']['data_paths'].values())]

		# create folder
		create_folder(self.wav_folders_audio_dataset + mfcc_paths + list(self.cfg['audio_dataset']['plot_paths'].values()))


	def create_my_recording_folders(self):
		"""
		create all necessary folders for my recordings
		"""

		# output path
		output_path = self.cfg['my_recordings']['out_path_root'] + self.param_path_my_recordings

		# create folder
		create_folder([self.cfg['my_recordings']['plot_path'], self.cfg['my_recordings']['wav_path'], output_path])	


	def create_ml_folders(self):
		"""
		create all necessary folders for ml
		"""

		# create folder
		create_folder(list(self.cfg['ml']['paths'].values()) + [self.model_path])



if __name__ == '__main__':
  """
  Path Collector
  """

  import yaml

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # init path collector
  path_coll = PathCollector(cfg)

  # some prints of variables
  print("param_path_audio_dataset: ", path_coll.param_path_audio_dataset)
  print("model_file: ", path_coll.model_file)