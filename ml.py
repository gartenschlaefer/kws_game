"""
Machine Learning file for training and evaluating the model with graphical collections and param savings
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import logging

# my stuff
from common import s_to_hms_str, create_folder, check_files_existance
from plots import plot_train_score, plot_confusion_matrix, plot_mfcc_only, plot_grid_images, plot_other_grid


class ML():
  """
  Machine Learning class
  """

  def __init__(self, cfg_ml, audio_dataset, batch_archive, net_handler, sub_model_path=None, encoder_label='', root_path='./'):

    # arguments
    self.cfg_ml = cfg_ml
    self.audio_dataset = audio_dataset
    self.batch_archive = batch_archive
    self.net_handler = net_handler
    self.sub_model_path = sub_model_path
    self.encoder_label = encoder_label
    self.root_path = root_path

    # paths
    self.paths = dict((k, self.root_path + v) for k, v in self.cfg_ml['paths'].items())

    # param path ml
    self.param_path_ml = 'bs-{}_it-{}_lr-{}/'.format(self.cfg_ml['train_params']['batch_size'], self.cfg_ml['train_params']['num_epochs'], str(self.cfg_ml['train_params']['lr']).replace('.', 'p'))

    # model path
    self.model_path = self.paths['model'] + self.cfg_ml['nn_arch'] + '/' + self.audio_dataset.param_path + self.param_path_ml

    # sub dir
    if self.sub_model_path is not None: self.model_path = self.model_path + sub_model_path

    # new sub dir for encoder label
    if len(self.encoder_label): self.model_path = self.model_path + encoder_label + '/'

    # model path folders
    self.model_path_folders = dict((k, self.model_path + v) for k, v in self.cfg_ml['model_path_folders'].items())

    # model file
    self.model_files = [self.model_path + model_name + '_' + self.cfg_ml['model_file_name'] for model_name, v in net_handler.models.items()]
    self.model_pre_files = [self.paths['model_pre'] + model_name + '_' + '{}_c-{}.pth'.format(self.cfg_ml['nn_arch'], self.batch_archive.n_classes) for model_name, v in net_handler.models.items()]
    
    # encoder decoder available
    enc, dec = net_handler.get_nn_arch_has_conv_coder()
    self.encoder_model_file = self.model_path + self.cfg_ml['encoder_model_file_name'] if enc else None
    self.decoder_model_file = self.model_path + self.cfg_ml['decoder_model_file_name'] if dec else None

    # params and metrics files
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
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('matplotlib.colorbar').disabled = True
    logging.getLogger('matplotlib.animation').disabled = True

    # load pre trained model
    if self.cfg_ml['load_pre_model']:
      self.net_handler.load_models(model_files=self.model_pre_files)


  def get_encoder_decoder_file_from_nn_arch(self):
    """
    check architecture on encoder or decoder file
    """

    # encoder available
    if self.net_handler.nn_arch in ['conv-encoder', 'conv-encoder-fc1', 'conv-encoder-fc3', 'conv-lim-encoder', 'conv-latent', 'adv-experimental', 'adv-experimental3', 'adv-collected-encoder', 'adv-lim-encoder', 'adv-lim-encoder-6']: 
      self.encoder_model_file = self.model_path + self.cfg_ml['encoder_model_file_name']

    # decoder available
    if self.net_handler.nn_arch in ['adv-experimental', 'adv-experimental3', 'adv-collected-encoder', 'adv-lim-encoder', 'adv-lim-encoder-6']:
      self.decoder_model_file = self.model_path + self.cfg_ml['decoder_model_file_name']


  def train(self, new_train_params=None, log_on=True, name_ext='', save_models=True):
    """
    training
    """

    # check if model already exists
    if check_files_existance([self.encoder_model_file, self.decoder_model_file] + self.model_files) and not self.cfg_ml['retrain']:

      # load model and check if it was possible
      if self.net_handler.load_models(model_files=self.model_files): return

    # if train params need to be changed
    train_params = self.cfg_ml['train_params'] if new_train_params is None else new_train_params

    # train
    train_score = self.net_handler.train_nn(train_params=train_params, batch_archive=self.batch_archive, callback_f=self.image_collect)

    # training info
    if log_on: logging.info('Training on arch: [{}], audio set param string: [{}], train_params: {}, device: [{}], time: {}'.format(self.cfg_ml['nn_arch'], self.audio_dataset.param_path, self.cfg_ml['train_params'], self.net_handler.device, s_to_hms_str(train_score.time_usage)))
    
    # save models and params
    if save_models:
      self.net_handler.save_models(model_files=self.model_files, encoder_model_file=self.encoder_model_file, decoder_model_file=self.decoder_model_file)
      self.save_params()
      self.save_metrics(train_score=train_score)

    # save infos
    self.save_infos()

    # save as pre trained model
    if self.cfg_ml['save_as_pre_model']:
      self.net_handler.save_models(model_files=self.model_pre_files)

    # plot score
    plot_train_score(train_score, plot_path=self.model_path, name_ext=name_ext)


  def save_params(self):
    """
    save parameter file
    """
    np.savez(self.params_file, nn_arch=self.net_handler.nn_arch, train_params=self.cfg_ml['train_params'], class_dict=self.batch_archive.class_dict, data_size=self.net_handler.data_size, feature_params=self.audio_dataset.feature_params)


  def save_metrics(self, train_score=None):
    """
    save training metrics
    """
    np.savez(self.metrics_file, train_score=train_score)


  def save_infos(self):
    """
    save some infos to a text file, e.g. model structure
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
    eval_score = self.net_handler.eval_nn(eval_set='test', batch_archive=self.batch_archive, collect_things=True, verbose=False)

    # score print of collected
    eval_score.info_collected(self.net_handler.nn_arch, self.audio_dataset.param_path, self.cfg_ml['train_params'], info_file=self.score_file, do_print=False)

    # log to file
    if self.cfg_ml['logging_enabled']: logging.info(eval_score.info_detail_log(self.net_handler.nn_arch, self.audio_dataset.param_path, self.cfg_ml['train_params']))

    # print confusion matrix
    print("confusion matrix:\n{}\n".format(eval_score.cm))

    # plot confusion matrix
    plot_confusion_matrix(eval_score.cm, self.batch_archive.classes, plot_path=self.model_path, name='confusion_test')


    # --
    # evaluation on my set

    if self.batch_archive.x_my is None:
      if self.cfg_ml['logging_enabled']: logging.info('\n')
      return

    print("\n--Evaluation on My Set:")

    # evaluation of model
    eval_score = self.net_handler.eval_nn(eval_set='my', batch_archive=self.batch_archive, collect_things=True, verbose=True)
    
    # score print of collected
    eval_score.info_collected(self.net_handler.nn_arch, self.audio_dataset.param_path, self.cfg_ml['train_params'], info_file=self.score_file, do_print=False)

    # log to file
    if self.cfg_ml['logging_enabled']: logging.info(eval_score.info_detail_log(self.net_handler.nn_arch, self.audio_dataset.param_path, self.cfg_ml['train_params']))

    # confusion matrix
    print("confusion matrix:\n{}\n".format(eval_score.cm))

    # plot confusion matrix
    plot_confusion_matrix(eval_score.cm, self.batch_archive.classes, plot_path=self.model_path, name='confusion_my')

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

          # context for colormap
          if k.find('layers.0.weight') != -1: context = 'weight0'
          elif k.find('layers.1.weight') != -1: context = 'weight1'
          else: context = 'weight0'

          # detach weights
          v = v.detach().cpu()

          # info
          #print("{} analyze: {}".format(k, v.shape))
          
          # plot images
          plot_grid_images(x=v.numpy(), context=context, color_balance=True, padding=1, num_cols=np.clip(v.shape[0], 1, 8), title=k + self.param_path_ml.replace('/', ' ') + ' ' + self.encoder_label + name_ext, plot_path=self.model_path, name=k + name_ext, show_plot=False)
          plot_grid_images(x=v.numpy(), context=context+'-div', color_balance=True, padding=1, num_cols=np.clip(v.shape[0], 1, 8), title=k + self.param_path_ml.replace('/', ' ') + ' ' + self.encoder_label + name_ext, plot_path=self.model_path_folders['diff_plots'], name='div_' + k + name_ext, show_plot=False)
          #plot_other_grid(x=v.numpy(), title=k + self.param_path_ml.replace('/', ' '), plot_path=self.model_path, name=k, show_plot=False)

    # generate samples from trained model (only for adversarial)
    fakes = self.net_handler.generate_samples(num_samples=30, to_np=True)
    if fakes is not None:
      plot_grid_images(x=fakes, context='mfcc', padding=1, num_cols=5, title='generated samples ' + self.encoder_label + name_ext, plot_path=self.model_path, name='generated_samples_' + self.encoder_label + name_ext, show_plot=False)
      plot_mfcc_only(fakes[0, 0], fs=16000, hop=160, plot_path=self.model_path, name='generated_sample_' + self.encoder_label + name_ext, show_plot=False)

    # animation (for generative networks)
    if self.cfg_ml['plot_animation']: self.create_anim()


  def image_collect(self, x):
    """
    collect images of mfcc's (used as callback function in the training of adversarial networks)
    """

    # append image
    self.img_list.append(x[0])


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

    # plot
    fig = plt.figure(figsize=(8,8))
    #plt.axis("off")

    # animation
    ani = animation.ArtistAnimation(fig, [[plt.imshow(i[0, :], animated=True)] for i in self.img_list], interval=1000, repeat_delay=1000, blit=True)

    # save
    ani.save("{}anim.mp4".format(self.model_path))

    plt.show()



def label_train_conv_encoder(cfg, all_feature_files, num_label_split, model_path, data_size):
  """
  label train
  """

  # encoder model list
  encoder_models = torch.nn.ModuleList()
  decoder_models = torch.nn.ModuleList()

  # split lim
  num_label_split = len(cfg['datasets']['speech_commands']['sel_labels']) if num_label_split > len(cfg['datasets']['speech_commands']['sel_labels']) else num_label_split
  
  # label separation
  label_split = np.array_split(cfg['datasets']['speech_commands']['sel_labels'], num_label_split)

  # conv encoder for classes
  for l_split in label_split:

    # concatenate splits for info
    l = '_'.join(l_split)

    # info
    print("\nconv encoder for: ", l)
    encoder_model, decoder_model = None, None

    # coder filse
    coder_files = [model_path + l + '/' + p for p in [cfg['ml']['encoder_model_file_name'], cfg['ml']['decoder_model_file_name']]]

    # check if encoder file exists
    if check_files_existance(coder_files):

      print("files found, load...")
      
      # create net handler
      net_handler = NetHandler(nn_arch=cfg['ml']['adv_params']['nn_archs_label'][-1], class_dict={l:0}, data_size=data_size, use_cpu=cfg['ml']['use_cpu'])

      # load models
      net_handler.models['d'].conv_encoder.load_state_dict(torch.load(coder_files[0]))
      net_handler.models['g'].conv_decoder.load_state_dict(torch.load(coder_files[1]))

      # add encoder model
      encoder_models.append(net_handler.models['d'].conv_encoder)
      decoder_models.append(net_handler.models['g'].conv_decoder)

      continue


    # batch archive for one label
    batch_archive = SpeechCommandsBatchArchive(all_feature_files, batch_size=cfg['ml']['train_params']['batch_size'])

    # iterations for learning conv encoders
    for i in range(cfg['ml']['adv_params']['num_iterations_label']):
      
      # go through selected archs
      for j, nn_arch in enumerate(cfg['ml']['adv_params']['nn_archs_label']):

        # reduce to single labels
        batch_archive.reduce_to_labels(l_split)

        # add noise data for conv-exp
        #if nn_arch in ['conv-experimental', 'adv-experimental3']:
        if nn_arch in ['conv-experimental']:
          #batch_archive.one_against_all(l, others_label='other', shuffle=True)
          batch_archive.add_noise_data(shuffle=True)

        # check classes
        print("classes: ", batch_archive.classes)

        # net handler
        net_handler = NetHandler(nn_arch=nn_arch, class_dict=batch_archive.class_dict, data_size=batch_archive.data_size, encoder_model=encoder_model, decoder_model=decoder_model, use_cpu=cfg['ml']['use_cpu'])

        # ml
        ml = ML(cfg_ml=cfg['ml'], audio_dataset=audio_set1, batch_archive=batch_archive, net_handler=net_handler, sub_model_path=cfg['ml']['conv_folder'], encoder_label=l)

        # change train params
        train_params = cfg['ml']['train_params'].copy()
        train_params['num_epochs'] = cfg['ml']['adv_params']['num_epochs_label']

        # train and analyze
        name_ext = '_{}-{}_{}'.format(i, j, nn_arch)
        ml.train(new_train_params=train_params, log_on=False, name_ext=name_ext, save_models=(i==cfg['ml']['adv_params']['num_iterations_label']-1))
        ml.analyze(name_ext=name_ext)

        # update encoder models
        if nn_arch == 'conv-experimental': encoder_model, decoder_model = net_handler.models['cnn'], None
        elif nn_arch in ['adv-experimental', 'adv-experimental3']: encoder_model, decoder_model = net_handler.models['d'], net_handler.models['g']

    # add encoder model
    encoder_models.append(encoder_model.conv_encoder)
    decoder_models.append(decoder_model.conv_decoder)

  return encoder_models, decoder_models


def get_adversarial_pair_arch(nn_arch):
  """
  adversarial pair architecture for adversarial conv encoder training
  """
  if nn_arch in ['conv-lim-encoder']: return 'adv-lim-encoder'
  elif nn_arch in ['conv-encoder-fc1', 'conv-encoder-fc3']: return 'adv-lim-encoder-6'
  return 'adv-collected-encoder'


def train_conv_encoders(cfg, audio_set1, all_feature_files, encoder_model=None, decoder_model=None):
  """
  train convolutional encoders
  """

  # check architecture
  if cfg['ml']['nn_arch'] not in ['conv-encoder', 'conv-lim-encoder', 'conv-latent', 'conv-encoder-fc1', 'conv-encoder-fc3']: return None, None

  # batch archive
  batch_archive = SpeechCommandsBatchArchive(all_feature_files, batch_size=cfg['ml']['train_params']['batch_size'])

  # nethandler
  net_handler = NetHandler(nn_arch=get_adversarial_pair_arch(cfg['ml']['nn_arch']), class_dict=batch_archive.class_dict, data_size=batch_archive.data_size, encoder_model=encoder_model, decoder_model=decoder_model, use_cpu=cfg['ml']['use_cpu'])
  
  # ml
  ml = ML(cfg_ml=cfg['ml'], audio_dataset=audio_set1, batch_archive=batch_archive, net_handler=net_handler, sub_model_path=cfg['ml']['conv_folder'])


  # check if encoder file exists
  if check_files_existance([ml.encoder_model_file, ml.decoder_model_file]):
    
    # load models
    net_handler.models['d'].conv_encoder.load_state_dict(torch.load(ml.model_path + cfg['ml']['encoder_model_file_name']))
    net_handler.models['g'].conv_decoder.load_state_dict(torch.load(ml.model_path + cfg['ml']['decoder_model_file_name']))

    # decoder weights only
    if cfg['ml']['adv_params']['use_decoder_weights']: return None, net_handler.models['g']

    # encoder weights only
    return net_handler.models['d'], None


  # train each label separate (used for collected net)
  if cfg['ml']['adv_params']['label_train']:

    # number of labels split
    num_label_split = net_handler.models['d'].conv_encoder.n_feature_maps[0][1] // 8

    # label train
    encoder_models, decoder_models = label_train_conv_encoder(cfg, all_feature_files, num_label_split, model_path=ml.model_path, data_size=batch_archive.data_size)

    # transfer coder weights
    net_handler.models['d'].conv_encoder.transfer_conv_weights_label_coders(encoder_models)
    net_handler.models['g'].conv_decoder.transfer_conv_weights_label_coders(decoder_models)

    ml.analyze(name_ext='_0-0_pre_norm')

    # normalize if requested
    if cfg['ml']['adv_params']['norm_label_weights']:
      for encoder_model, decoder_model in zip(encoder_models, decoder_models):
        with torch.no_grad():
          encoder_model.conv_layers[0].weight.div_(torch.norm(encoder_model.conv_layers[0].weight, keepdim=True))
          encoder_model.conv_layers[1].weight.div_(torch.norm(encoder_model.conv_layers[1].weight, keepdim=True))
          decoder_model.deconv_layers[0].weight.div_(torch.norm(decoder_model.deconv_layers[0].weight, keepdim=True))
          decoder_model.deconv_layers[1].weight.div_(torch.norm(decoder_model.deconv_layers[1].weight, keepdim=True))

      # transfer coder weights
      net_handler.models['d'].conv_encoder.transfer_conv_weights_label_coders(encoder_models)
      net_handler.models['g'].conv_decoder.transfer_conv_weights_label_coders(decoder_models)

      ml.analyze(name_ext='_0-1_post_norm')

    # # add noise to each weight
    # with torch.no_grad():
    #   for param in encoder_model.parameters():
    #     param.add_(torch.randn(param.shape) * 0.01)

    #torch.nn.init.xavier_uniform_(collected_encoder_model.conv_layers[1].weight, gain=torch.nn.init.calculate_gain('relu'))


  # pre training
  if cfg['ml']['adv_params']['pre_train']:

    # set epochs
    train_params = cfg['ml']['train_params'].copy()
    train_params['num_epochs'] = cfg['ml']['adv_params']['num_epochs_pre']

    # training
    ml.analyze(name_ext='_1_pre-adv')
    ml.train(new_train_params=train_params, log_on=False, name_ext='_adv-post_train', save_models=True)
    ml.analyze(name_ext='_2_post-adv')


  # return decoder model
  if cfg['ml']['adv_params']['use_decoder_weights']: return None, net_handler.models['g']

  # return encoder model
  return net_handler.models['d'], None


def get_audiosets(cfg):
  """
  get audioset
  """

  # channel size
  channel_size = 1 if not cfg['feature_params']['use_channels'] else int(cfg['feature_params']['use_cepstral_features']) + int(cfg['feature_params']['use_delta_features']) +  int(cfg['feature_params']['use_double_delta_features'])

  # feature size
  feature_size = (cfg['feature_params']['n_ceps_coeff'] + int(cfg['feature_params']['use_energy_features'])) * int(cfg['feature_params']['use_cepstral_features']) + (cfg['feature_params']['n_ceps_coeff'] + int(cfg['feature_params']['use_energy_features'])) * int(cfg['feature_params']['use_delta_features']) + (cfg['feature_params']['n_ceps_coeff'] + int(cfg['feature_params']['use_energy_features'])) * int(cfg['feature_params']['use_double_delta_features']) if not cfg['feature_params']['use_channels'] else (cfg['feature_params']['n_ceps_coeff'] + int(cfg['feature_params']['use_energy_features']))
  
  # exception
  if feature_size == 0 or channel_size == 0: return None, None, None

  # audio sets
  audio_set1 = AudioDataset(cfg['datasets']['speech_commands'], cfg['feature_params'])
  audio_set2 = AudioDataset(cfg['datasets']['my_recordings'], cfg['feature_params'])

  # create dataset if not existing
  if not check_files_existance(audio_set1.feature_files):
    audio_set1 = SpeechCommandsDataset(cfg['datasets']['speech_commands'], feature_params=cfg['feature_params'], verbose=False)
    audio_set1.extract_features()

  # create dataset if not existing
  if not check_files_existance(audio_set2.feature_files):
    audio_set2 = MyRecordingsDataset(cfg['datasets']['my_recordings'], feature_params=cfg['feature_params'], verbose=False)
    audio_set2.extract_features()

  # select feature files
  all_feature_files = audio_set1.feature_files + audio_set2.feature_files if len(audio_set1.labels) == len(audio_set2.labels) else audio_set1.feature_files

  return audio_set1, audio_set2, all_feature_files


def cfg_changer(cfg_file):
  """
  change config for more convenient training
  """

  cfg = yaml.safe_load(open(cfg_file))

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
  ML - Machine Learning file
  """

  import yaml
  import torch

  from conv_nets import ConvEncoder, ConvDecoder
  from batch_archive import SpeechCommandsBatchArchive
  from net_handler import NetHandler
  from audio_dataset import AudioDataset, SpeechCommandsDataset, MyRecordingsDataset
  from test_bench import TestBench


  # go through configs
  for cfg in cfg_changer(cfg_file='./config.yaml'):
    
    # get audio sets
    audio_set1, audio_set2, all_feature_files = get_audiosets(cfg)

    # skip condition
    if audio_set1 is None or all_feature_files is None: 
      print("skip config")
      continue

    # encoder, decoder models for certain architectures necessary
    encoder_model, decoder_model = None, None

    # conv encoder training
    encoder_model, decoder_model = train_conv_encoders(cfg, audio_set1, all_feature_files)


    # create batch archive
    batch_archive = SpeechCommandsBatchArchive(all_feature_files, batch_size=cfg['ml']['train_params']['batch_size'])

    # adversarial training
    if cfg['ml']['nn_arch'] in ['adv-experimental']:
      batch_archive.reduce_to_label('up')
      #batch_archive.add_noise_data(shuffle=True)
      #batch_archive.one_against_all('up', others_label='other', shuffle=True)

    # net handler
    net_handler = NetHandler(nn_arch=cfg['ml']['nn_arch'], class_dict=batch_archive.class_dict, data_size=batch_archive.data_size, encoder_model=encoder_model, decoder_model=decoder_model, use_cpu=cfg['ml']['use_cpu'])


    # --
    # ML

    # instance
    ml = ML(cfg_ml=cfg['ml'], audio_dataset=audio_set1, batch_archive=batch_archive, net_handler=net_handler)

    # run ml stuff
    ml.analyze(name_ext='_init'), ml.train(), ml.eval(), ml.analyze()


    # --
    # Test Bench

    # create test bench
    test_bench = TestBench(cfg['test_bench'], test_model_path=ml.model_path)

    # shift invariance test
    test_bench.test_invariances()




