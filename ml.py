"""
machine learning
"""

import numpy as np
import os
import logging

from glob import glob

# my stuff
from common import s_to_hms_str, create_folder, check_files_existance
from plots import plot_train_score, plot_confusion_matrix, plot_mfcc_only, plot_grid_images, plot_mfcc_anim


class ML():
  """
  machine learning class
  """

  def __init__(self, cfg_ml, audio_dataset, audio_dataset_my=None, nn_arch_context='config', encoder_label='', encoder_model=None, decoder_model=None, root_path='./'):

    # arguments
    self.cfg_ml = cfg_ml
    self.audio_dataset = audio_dataset
    self.audio_dataset_my = audio_dataset_my
    self.nn_arch_context = nn_arch_context
    #self.sub_model_path = sub_model_path
    self.encoder_label = encoder_label
    self.encoder_model = encoder_model
    self.decoder_model = decoder_model
    self.root_path = root_path

    # get architecture for net handler
    self.nn_arch = self.get_net_handler_arch()

    # train parameters
    self.train_params = self.get_training_params()

    # create batch archive
    self.batch_archive = SpeechCommandsBatchArchive(feature_file_dict={**self.audio_dataset.feature_file_dict, **self.audio_dataset_my.feature_file_dict}, batch_size_dict={'train': self.train_params['batch_size'], 'test': 5, 'validation': 5, 'my': 1}, shuffle=True) if audio_dataset_my is not None else SpeechCommandsBatchArchive(feature_file_dict=self.audio_dataset.feature_file_dict, batch_size_dict={'train': self.train_params['batch_size'], 'test': 5, 'validation': 5, 'my': 1}, shuffle=True)

    # create batches
    self.batch_archive.create_batches() if not self.nn_arch.startswith('adv') else self.batch_archive.create_batches(selected_labels=['left'])

    # net handler
    self.net_handler = NetHandler(nn_arch=self.nn_arch, class_dict=self.batch_archive.class_dict, data_size=self.batch_archive.data_size, feature_params=self.audio_dataset.feature_params, encoder_model=self.encoder_model, decoder_model=self.decoder_model, use_cpu=self.cfg_ml['use_cpu'])


    # paths
    self.paths = dict((k, self.root_path + v) for k, v in self.cfg_ml['paths'].items())

    # adv pre train folder
    self.param_path_ml = self.cfg_ml['adv_pre_folder'] if self.nn_arch_context in ['adv_dual', 'adv_label', 'adv_label_all'] else ''

    # parameter path ml
    self.param_path_ml += 'bs-{}_it-{}_lr-{}'.format(self.train_params['batch_size'], self.train_params['num_epochs'], str(self.train_params['lr']).replace('.', 'p')) if not (self.nn_arch.startswith('adv') or self.nn_arch.startswith('hyb')) else 'bs-{}_it-{}_lr-d-{}_lr-g-{}'.format(self.train_params['batch_size'], self.train_params['num_epochs'], str(self.train_params['lr_d']).replace('.', 'p'), str(self.train_params['lr_g']).replace('.', 'p'))
    self.param_path_ml += '_dual' if self.cfg_ml['adv_params']['dual_train'] and self.nn_arch_context in ['adv_dual'] else ''
    self.param_path_ml += '_label' if self.cfg_ml['adv_params']['label_train'] and self.nn_arch_context in ['adv_label', 'adv_label_all'] else ''
    self.param_path_ml += '_pre-adv' if (self.cfg_ml['adv_params']['label_train'] or self.cfg_ml['adv_params']['dual_train']) and not self.nn_arch_context in ['adv_dual', 'adv_label', 'adv_label_all'] else '/'

    # model path
    self.model_path = self.paths['model'] + self.cfg_ml['nn_arch'] + '/' + self.audio_dataset.param_path + self.param_path_ml

    # model instance
    self.model_instance = '_i{}'.format(len(glob(self.model_path[:-1] + '*')))

    # create new instance
    self.model_path = self.model_path[:-1] + self.model_instance + '/' if os.path.isdir(self.model_path) and self.cfg_ml['new_instance'] and self.nn_arch_context not in ['adv_dual', 'adv_label', 'adv_label_all'] else self.model_path

    # new sub directory for encoder label
    self.model_path += self.encoder_label + '/' if len(self.encoder_label) else ''

    # create model path
    create_folder([self.model_path])

    # model path folders
    self.model_path_folders = dict((k, self.model_path + v) for k, v in self.cfg_ml['model_path_folders'].items())

    # model file
    self.model_files = [self.model_path + model_name + '_' + self.cfg_ml['model_file_name'] for model_name, v in self.net_handler.models.items()]
    self.model_pre_files = [self.model_path + model_name + '_' + self.cfg_ml['model_pre_file_name'] for model_name, v in self.net_handler.models.items()]

    # parameter and metrics files
    self.params_file = self.model_path + self.cfg_ml['params_file_name']
    self.metrics_file = self.model_path + self.cfg_ml['metrics_file_name']
    self.info_file = self.model_path + self.cfg_ml['info_file_name']
    self.score_file = self.model_path + self.cfg_ml['score_file_name']

    # image list (for adversarial)
    self.img_list = []

    # create ml folders
    create_folder(list(self.paths.values()) + [self.model_path] + list(self.model_path_folders.values()))

    # config
    logging.basicConfig(filename=self.paths['log'] + 'ml.log', level=logging.INFO, format='%(asctime)s %(message)s')

    # disable unwanted logs
    logging.getLogger('matplotlib.font_manager').disabled, logging.getLogger('matplotlib.colorbar').disabled, logging.getLogger('matplotlib.animation').disabled = True, True, True

    # load pre trained model
    if self.cfg_ml['load_pre_model']: self.net_handler.load_models(model_files=self.model_pre_files)


  def get_training_params(self):
    """
    training params for each architecture
    """

    # cnn networks
    if self.nn_arch.startswith('conv'): return self.cfg_ml['train_params']['cnn']

    # adversarial networks
    elif self.nn_arch.startswith('adv'):

      # pre training dual
      if self.nn_arch_context == 'adv_dual': return self.cfg_ml['train_params']['adv_dual']
      elif self.nn_arch_context in ['adv_label', 'adv_label_all']: return self.cfg_ml['train_params']['adv_label']
      return self.cfg_ml['train_params']['adv']

    # hybrid nets
    elif self.nn_arch.startswith('hyb'): return self.cfg_ml['train_params']['hyb']

    # wavenet
    elif self.nn_arch.startswith('wave'): return self.cfg_ml['train_params']['wave']

    return self.cfg_ml['train_params']['cnn']


  def get_net_handler_arch(self):
    """
    get nn arch for net handler
    """

    print(self.nn_arch_context)

    # from config (usual)
    if self.nn_arch_context == 'config': return self.cfg_ml['nn_arch']

    # get adv dual arch
    elif self.nn_arch_context in ['adv_dual', 'adv_label_all']: return self.get_adversarial_pair_arch(self.cfg_ml['nn_arch'], label_arch=False)

    # get adv label arch
    elif self.nn_arch_context == 'adv_label': return self.get_adversarial_pair_arch(self.cfg_ml['nn_arch'], label_arch=True)

    return self.cfg_ml['nn_arch']


  def get_adversarial_pair_arch(self, nn_arch, label_arch=False):
    """
    adversarial pair architecture for dual and label architecture
    """

    # jim network
    if nn_arch in ['conv-jim']: return 'adv-jim' if not label_arch else 'adv-jim-label'

    return 'adv-jim'


  def train(self, new_train_params=None, log_on=True, name_ext='', save_models_enabled=True):
    """
    training
    """

    # check if model already exists
    if check_files_existance(self.model_files) and not self.cfg_ml['retrain']:

      # load model and check if it was possible
      self.net_handler.load_models(model_files=self.model_files)

    # change training parameters if requested
    train_params = self.train_params if new_train_params is None else new_train_params

    # train
    train_score = self.net_handler.train_nn(train_params=train_params, batch_archive=self.batch_archive, callback_f=self.image_collect)

    # training info
    if log_on: logging.info('Training on arch: [{}], audio set param string: [{}], train_params: {}, device: [{}], time: {}'.format(self.cfg_ml['nn_arch'], self.audio_dataset.param_path, train_params, self.net_handler.device, s_to_hms_str(train_score.score_dict['time_usage'])))
    
    # save models and parameters
    if save_models_enabled:
      self.net_handler.save_models(model_files=self.model_files)
      self.save_params()
      self.save_metrics(train_score=train_score)

    # save info
    self.save_infos()

    # save as pre trained model
    if self.cfg_ml['save_as_pre_model']: self.net_handler.save_models(model_files=self.model_pre_files)

    # plot score
    plot_train_score(train_score.score_dict, plot_path=self.model_path, name_ext=name_ext)


  def save_params(self):
    """
    save parameter file
    """
    np.savez(self.params_file, nn_arch=self.net_handler.nn_arch, train_params=self.train_params, class_dict=self.batch_archive.class_dict, data_size=self.net_handler.data_size, feature_params=self.audio_dataset.feature_params)


  def save_metrics(self, train_score=None):
    """
    save training metrics
    """
    np.savez(self.metrics_file, train_score_dict=train_score.score_dict)


  def save_infos(self):
    """
    save some info to a text file, e.g. model structure
    """
    with open(self.info_file, 'w') as f:
      print(self.net_handler.models, file=f)
      print("\n", self.cfg_ml['adv_params'], file=f)
    

  def eval(self):
    """
    evaluation
    """

    print("\n--Evaluation on Test Set:")

    # evaluation of model
    eval_score = self.net_handler.eval_nn('test', batch_archive=self.batch_archive, collect_things=True, verbose=False)

    # score print of collected
    eval_score.info_collected(self.net_handler.nn_arch, self.audio_dataset.param_path, self.train_params, info_file=self.score_file, do_print=False)

    # log to file
    if self.cfg_ml['logging_enabled']: logging.info(eval_score.info_detail_log(self.net_handler.nn_arch, self.audio_dataset.param_path, self.train_params))

    # print confusion matrix
    print("confusion matrix:\n{}\n".format(eval_score.cm))

    # plot confusion matrix
    plot_confusion_matrix(eval_score.cm, self.batch_archive.class_dict.keys(), plot_path=self.model_path, name='confusion_test')


    # --
    # evaluation on my set

    # check if my set exists
    if not 'my' in self.batch_archive.set_names:
      if self.cfg_ml['logging_enabled']: logging.info('\n')
      return

    print("\n--Evaluation on My Set:")

    # evaluation of model
    eval_score = self.net_handler.eval_nn('my', batch_archive=self.batch_archive, collect_things=True, verbose=True)
    
    # score print of collected
    eval_score.info_collected(self.net_handler.nn_arch, self.audio_dataset.param_path, self.train_params, info_file=self.score_file, do_print=False)

    # log to file
    if self.cfg_ml['logging_enabled']: logging.info(eval_score.info_detail_log(self.net_handler.nn_arch, self.audio_dataset.param_path, self.train_params))

    # confusion matrix
    print("confusion matrix:\n{}\n".format(eval_score.cm))

    # plot confusion matrix
    plot_confusion_matrix(eval_score.cm, self.batch_archive.class_dict.keys(), plot_path=self.model_path, name='confusion_my')

    # new line for log
    if self.cfg_ml['logging_enabled']: logging.info('\n')


  def analyze(self, name_ext=''):
    """
    analyze function, e.g. analyze weights
    """

    # analyze weights
    for model_name, model in self.net_handler.models.items():

      # go through each weight
      for k, v in model.state_dict().items():

        # convolutional layers
        if k.find('conv') != -1 and not k.__contains__('bias'):

          # context for color map
          if k.find('layers.0.weight') != -1: context = 'weight0'
          elif k.find('layers.1.weight') != -1: context = 'weight1'
          else: context = 'weight0'

          # detach weights
          x = v.detach().cpu().numpy()

          # plot images
          plot_grid_images(x, context=context, color_balance=True, padding=1, num_cols=np.clip(v.shape[0], 1, 8), title=k + self.param_path_ml.replace('/', ' ') + ' ' + self.encoder_label + name_ext, plot_path=self.model_path_folders['conv_plots'], name=k + name_ext, show_plot=False)
          plot_grid_images(x, context=context+'-div', color_balance=True, padding=1, num_cols=np.clip(v.shape[0], 1, 8), title=k + self.param_path_ml.replace('/', ' ') + ' ' + self.encoder_label + name_ext, plot_path=self.model_path_folders['conv_diff_plots'], name='div_' + k + name_ext, show_plot=False)

    # generate samples from trained model (only for adversarial)
    fakes = self.net_handler.generate_samples(num_samples=30, to_np=True)
    if fakes is not None:
      plot_grid_images(x=fakes, context='mfcc', padding=1, num_cols=5, title='generated samples ' + self.encoder_label + name_ext, plot_path=self.model_path, name='generated_samples_' + self.encoder_label + name_ext, show_plot=False)
      plot_mfcc_only(fakes[0, 0], fs=16000, hop=160, plot_path=self.model_path, name='generated_sample_' + self.encoder_label + name_ext, show_plot=False)

    # animation (for generative networks)
    if self.cfg_ml['create_animation']: self.create_anim()

    # plot collections
    if self.cfg_ml['create_collections']: self.collections()


  def image_collect(self, x, epoch):
    """
    collect images of mfcc's (used as callback function in the training of adversarial networks)
    """

    # disabled feature
    if not self.cfg_ml['create_collections']: return

    # append image
    self.img_list.append(x)

    # model file checkpoints
    model_files = [self.model_path_folders['train_collections'] + p + '{:0>5}.pth'.format(epoch) for p in ['g_model_ep-', 'd_model_ep-']]

    # save state dictionary
    self.net_handler.save_models(model_files)


  def collections(self):
    """
    get collection of models and plot something
    """
    from glob import glob

    collection_files = glob(self.model_path_folders['train_collections'] + '*.pth')

    # get specific model files
    g_model_files = [f for f in collection_files if 'g_model' in f]
    d_model_files = [f for f in collection_files if 'd_model' in f]

    # load models
    g_d1 = [torch.load(g)['conv_decoder.deconv_layers.1.weight'] for g in g_model_files]
    d_c1 = [torch.load(d)['conv_encoder.conv_layers.0.weight'] for d in d_model_files]

    # plot models
    for i, x in enumerate(g_d1): plot_grid_images(x.cpu(), context='weight0', num_cols=8, color_balance=True, title='', plot_path=self.model_path_folders['train_collections'], name='g' + str(i))
    for i, x in enumerate(d_c1): plot_grid_images(x.cpu(), context='weight0', num_cols=8, color_balance=True, title='', plot_path=self.model_path_folders['train_collections'], name='d' + str(i))


  def create_anim(self):
    """
    create image animation
    """

    # empty check
    if not len(self.img_list):
      print("no images collected for animation")
      return

    # images for evaluation on training
    print("amount of mfccs for anim: ", len(self.img_list))

    # create anim
    plot_mfcc_anim(self.img_list, plot_path=self.model_path, name='mfcc-anim')



# --
# pre-train

def pre_train_adv_label(cfg, audio_set1, encoder_model=None, decoder_model=None):
  """
  adv-label train
  """

  # check architecture
  if cfg['ml']['nn_arch'] not in ['conv-jim', 'conv-encoder-fc1', 'conv-encoder-fc3']: return None, None

  # encoder model list
  encoder_models = torch.nn.ModuleList()
  decoder_models = torch.nn.ModuleList()

  # calculate label split
  num_label_split = 48 // 8

  # split lim
  num_label_split = len(cfg['datasets']['speech_commands']['sel_labels']) if num_label_split > len(cfg['datasets']['speech_commands']['sel_labels']) else num_label_split
  
  # label separation
  label_split = np.array_split(cfg['datasets']['speech_commands']['sel_labels'], num_label_split)

  # conv encoder for classes
  for l_split in label_split:

    # concatenate splits for info
    l_split_names = '_'.join(l_split)

    # info
    print("\nadv-pre model for: ", l_split_names)
    encoder_model, decoder_model = None, None

    # ml
    ml = ML(cfg_ml=cfg['ml'], audio_dataset=audio_set1, nn_arch_context='adv_label', encoder_label=l_split_names)

    # check if files exists
    if check_files_existance(ml.model_files):

      print("files found, load...")
      
      # load models
      ml.net_handler.load_models(ml.model_files)

      # add encoder and decoder
      encoder_models.append(ml.net_handler.models['d']), decoder_models.append(ml.net_handler.models['g'])

      continue

    # create batches
    ml.batch_archive.create_batches(l_split)

    # training
    ml.analyze(name_ext='_1-0_pre-adv')
    ml.train(log_on=False, name_ext='_adv-post_train', save_models_enabled=True)
    ml.analyze(name_ext='_1-1_post-adv')

    # add encoder and decoder
    encoder_models.append(ml.net_handler.models['d']), decoder_models.append(ml.net_handler.models['g'])


  # ml dual network
  ml_all = ML(cfg_ml=cfg['ml'], audio_dataset=audio_set1, nn_arch_context='adv_label_all')

  # transfer coder weights
  ml_all.analyze(name_ext='_init')
  ml_all.net_handler.models['d'].transfer_params_label_models(encoder_models)
  ml_all.net_handler.models['g'].transfer_params_label_models(decoder_models)
  ml_all.analyze()
  ml_all.net_handler.save_models(ml_all.model_files)

  return (None, ml_all.net_handler.models['g']) if cfg['ml']['adv_params']['use_decoder_weights'] else (ml_all.net_handler.models['d'], None)


def pre_train_adv_dual(cfg, audio_set1, encoder_model=None, decoder_model=None):
  """
  train convolutional encoders
  """

  # check architecture
  if cfg['ml']['nn_arch'] not in ['conv-jim', 'conv-encoder-fc1', 'conv-encoder-fc3']: return None, None

  # ml dual network
  ml = ML(cfg_ml=cfg['ml'], audio_dataset=audio_set1, nn_arch_context='adv_dual')

  # check if encoder file exists
  if check_files_existance(ml.model_files):
    
    # load models
    ml.net_handler.load_models(ml.model_files)

    # return encoder or decoder model
    return (None, ml.net_handler.models['g']) if cfg['ml']['adv_params']['use_decoder_weights'] else (ml.net_handler.models['d'], None)

  # create batches with all labels
  ml.batch_archive.create_batches()

  # training
  ml.analyze(name_ext='_1-0_pre-adv')
  ml.train(log_on=False, name_ext='_adv-post_train', save_models_enabled=True)
  ml.analyze(name_ext='_1-1_post-adv')

  # return either encoder or decoder model
  return (None, ml.net_handler.models['g']) if cfg['ml']['adv_params']['use_decoder_weights'] else (ml.net_handler.models['d'], None)


def get_audiosets(cfg):
  """
  get audioset
  """

  # channel size
  channel_size = 1 if not cfg['feature_params']['use_channels'] else int(cfg['feature_params']['use_cepstral_features']) + int(cfg['feature_params']['use_delta_features']) +  int(cfg['feature_params']['use_double_delta_features'])

  # feature size
  feature_size = (cfg['feature_params']['n_ceps_coeff'] + int(cfg['feature_params']['use_energy_features'])) * int(cfg['feature_params']['use_cepstral_features']) + (cfg['feature_params']['n_ceps_coeff'] + int(cfg['feature_params']['use_energy_features'])) * int(cfg['feature_params']['use_delta_features']) + (cfg['feature_params']['n_ceps_coeff'] + int(cfg['feature_params']['use_energy_features'])) * int(cfg['feature_params']['use_double_delta_features']) if not cfg['feature_params']['use_channels'] else (cfg['feature_params']['n_ceps_coeff'] + int(cfg['feature_params']['use_energy_features']))
  
  # exception
  if feature_size == 0 or channel_size == 0: return None, None

  # audio sets
  audio_set1 = AudioDataset(cfg['datasets']['speech_commands'], cfg['feature_params'])
  audio_set2 = AudioDataset(cfg['datasets']['my_recordings'], cfg['feature_params'])

  # create dataset if not existing
  if not check_files_existance(audio_set1.feature_file_dict.values()):
    audio_set1 = SpeechCommandsDataset(cfg['datasets']['speech_commands'], feature_params=cfg['feature_params'])
    audio_set1.extract_features()

  # create dataset if not existing
  if not check_files_existance(audio_set2.feature_file_dict.values()):
    audio_set2 = MyRecordingsDataset(cfg['datasets']['my_recordings'], feature_params=cfg['feature_params'])
    audio_set2.extract_features()

  return audio_set1, audio_set2


def cfg_changer(cfg_file):
  """
  change config for more convenient training
  """

  # load config
  cfg = yaml.safe_load(open(cfg_file))

  # change config upon nn arch
  cfg['feature_params']['use_mfcc_features'] = False if cfg['ml']['nn_arch'] == 'wavenet' else True

  # no config changes allowed
  if not cfg['config_changer_allowed']: return [cfg]

  # cfg list and selection
  cfg_list, binary4 = [], [[bool(int(b)) for b in np.binary_repr(i, width=4)] for i in np.arange(16)]

  # all permutations
  for binary in binary4:

    # load cofig
    cfg = yaml.safe_load(open(cfg_file))

    # change config
    cfg['feature_params']['use_cepstral_features'], cfg['feature_params']['use_delta_features'], cfg['feature_params']['use_double_delta_features'], cfg['feature_params']['use_energy_features'] = binary
    
    # append to list
    cfg_list.append(cfg)

  return cfg_list


if __name__ == '__main__':
  """
  machine learning main
  """

  import yaml
  import torch

  from audio_dataset import AudioDataset, SpeechCommandsDataset, MyRecordingsDataset
  from batch_archive import SpeechCommandsBatchArchive
  from net_handler import NetHandler
  from test_bench import TestBench


  # go through configs
  for cfg in cfg_changer(cfg_file='./config.yaml'):
    
    # get audio sets
    audio_set1, audio_set2 = get_audiosets(cfg)

    # skip condition
    if audio_set1 is None: 
      print("skip config")
      continue


    # adversarial pre training with dual network
    (encoder_model, decoder_model) = pre_train_adv_dual(cfg, audio_set1, encoder_model=None, decoder_model=None) if cfg['ml']['adv_params']['dual_train'] else (None, None)
    
    # adversarial pre training with label networks
    (encoder_model, decoder_model) = pre_train_adv_label(cfg, audio_set1, encoder_model=None, decoder_model=None) if cfg['ml']['adv_params']['label_train'] else (encoder_model, decoder_model)


    # --
    # machine learning and evaluation

    for i in range(cfg['ml']['num_instances']):

      # instance
      ml = ML(cfg_ml=cfg['ml'], audio_dataset=audio_set1, audio_dataset_my=audio_set2, encoder_model=encoder_model, decoder_model=decoder_model)

      # analyze and train eval analyze
      ml.analyze(name_ext='_init')
      ml.train()
      ml.eval()
      ml.analyze()


      # --
      # test bench

      # only for non adversarials
      if not cfg['ml']['nn_arch'].startswith('adv-'):

        # create test bench
        test_bench = TestBench(cfg['test_bench'], test_model_path=ml.model_path)

        # shift invariance test
        test_bench.test_invariances()