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
    if self.sub_model_path is not None and len(self.encoder_label) is not None: self.model_path = self.model_path + sub_model_path + '_' + encoder_label + '/'

    # model path folders
    self.model_path_folders = dict((k, self.model_path + v) for k, v in self.cfg_ml['model_path_folders'].items())

    # model file
    self.model_files = [self.model_path + model_name + '_' + self.cfg_ml['model_file_name'] for model_name, v in net_handler.models.items()]
    self.model_pre_files = [self.paths['model_pre'] + model_name + '_' + '{}_c-{}.pth'.format(self.cfg_ml['nn_arch'], self.batch_archive.n_classes) for model_name, v in net_handler.models.items()]
    
    # encoder model file
    self.encoder_model_file, self.decoder_model_file = None, None
    if self.cfg_ml['nn_arch'] in ['conv-encoder'] and len(self.encoder_label): self.encoder_model_file = self.model_path + self.cfg_ml['encoder_model_file_name']
    if self.cfg_ml['nn_arch'] in ['conv-encoder'] and len(self.encoder_label): self.decoder_model_file = self.model_path + self.cfg_ml['decoder_model_file_name']

    # params and metrics files
    self.params_file = self.model_path + self.cfg_ml['params_file_name']
    self.metrics_file = self.model_path + self.cfg_ml['metrics_file_name']
    self.info_file = self.model_path + self.cfg_ml['info_file_name']

    # image list (for adversarial)
    self.img_list = []

    # create ml folders
    create_folder(list(self.paths.values()) + [self.model_path] + list(self.model_path_folders.values()))

    # config
    logging.basicConfig(filename=self.paths['log'] + 'ml.log', level=logging.DEBUG, format='%(asctime)s %(message)s')

    # disable unwanted logs
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('matplotlib.colorbar').disabled = True
    logging.getLogger('matplotlib.animation').disabled = True

    # load pre trained model
    if self.cfg_ml['load_pre_model']:
      self.net_handler.load_models(model_files=self.model_pre_files)


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
    if log_on: logging.info('Traning on arch: [{}], train_params: {}, device: [{}], time: {}'.format(self.cfg_ml['nn_arch'], self.cfg_ml['train_params'], self.net_handler.device, s_to_hms_str(train_score.time_usage)))
    
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
    

  def eval(self):
    """
    evaluation
    """

    print("\n--Evaluation on Test Set:")

    # evaluation of model
    eval_score = self.net_handler.eval_nn(eval_set='test', batch_archive=self.batch_archive, calc_cm=True, verbose=False)

    # print accuracy
    eval_log = eval_score.info_log(do_print=False)

    # log to file
    if self.cfg_ml['logging_enabled']:
      logging.info(eval_log)

    # print confusion matrix
    print("confusion matrix:\n{}\n".format(eval_score.cm))

    # plot confusion matrix
    plot_confusion_matrix(eval_score.cm, self.batch_archive.classes, plot_path=self.model_path, name='confusion_test')


    # --
    # evaluation on my set

    if self.batch_archive.x_my is not None:

      print("\n--Evaluation on My Set:")

      # evaluation of model
      eval_score = self.net_handler.eval_nn(eval_set='my', batch_archive=self.batch_archive, calc_cm=True, verbose=True)
      print("confusion matrix:\n{}\n".format(eval_score.cm))

      # plot confusion matrix
      plot_confusion_matrix(eval_score.cm, self.batch_archive.classes, plot_path=self.model_path, name='confusion_my')


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
          print("{} analyze: {}".format(k, v.shape))
          
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



def train_conv_encoders(cfg, audio_set1, all_feature_files, encoder_model=None, decoder_model=None):
  """
  train convolutional encoders
  """

  import torch
  from conv_nets import ConvEncoder, ConvDecoder


  # # batch archive
  # batch_archive = SpeechCommandsBatchArchive(all_feature_files, batch_size=cfg['ml']['train_params']['batch_size'])

  # # dummy net handler
  # net_handler = NetHandler(nn_arch='none', class_dict=batch_archive.class_dict, data_size=batch_archive.data_size)

  # # collected encoder
  # ml_collected = ML(cfg_ml=cfg['ml'], audio_dataset=audio_set1, batch_archive=batch_archive, net_handler=net_handler)


  # batch archive
  batch_archive = SpeechCommandsBatchArchive(all_feature_files, batch_size=cfg['ml']['train_params']['batch_size'])

  # net handler for pre adv training
  if cfg['ml']['nn_arch'] in ['conv-lim-encoder']:
    net_handler = NetHandler(nn_arch='adv-lim-encoder', class_dict=batch_archive.class_dict, data_size=batch_archive.data_size, encoder_model=encoder_model, decoder_model=decoder_model, use_cpu=cfg['ml']['use_cpu'])
  else:
    net_handler = NetHandler(nn_arch='adv-collected-encoder', class_dict=batch_archive.class_dict, data_size=batch_archive.data_size, encoder_model=encoder_model, decoder_model=decoder_model, use_cpu=cfg['ml']['use_cpu'])

  # ml
  ml = ML(cfg_ml=cfg['ml'], audio_dataset=audio_set1, batch_archive=batch_archive, net_handler=net_handler)


  # coders classes
  #net_class = 'lim-encoder' if cfg['ml']['nn_arch'] in ['conv-lim-encoder'] else 'label-collect-encoder'

  # collected encoder/decoder models
  #collected_encoder_model = ConvEncoder(batch_archive.n_classes, batch_archive.data_size, net_class=net_class)
  #collected_decoder_model = ConvDecoder(batch_archive.n_classes, batch_archive.data_size, n_latent=100, net_class=net_class)

  # check if encoder file exists
  if check_files_existance([ml.model_path + p for p in [cfg['ml']['encoder_model_init_file_name'], cfg['ml']['decoder_model_init_file_name']]]):
    
    # load models
    #collected_encoder_model.load_state_dict(torch.load(ml.model_path + cfg['ml']['encoder_model_init_file_name']))
    #collected_decoder_model.load_state_dict(torch.load(ml.model_path + cfg['ml']['decoder_model_init_file_name']))

    # # add noise to each weight
    # with torch.no_grad():
    #   for param in encoder_model.parameters():
    #     param.add_(torch.randn(param.shape) * 0.01)
    # torch.nn.init.xavier_uniform_(collected_encoder_model.conv_layers[1].weight, gain=torch.nn.init.calculate_gain('relu'))

    # print("encoder / decoder model exists and loaded")
    # # use decoder
    # if cfg['ml']['adv_params']['use_decoder_weights']:
    #   collected_encoder_model.transfer_decoder_weights(collected_decoder_model)
    # return collected_encoder_model

    # load models
    net_handler.models['d'].conv_encoder.load_state_dict(torch.load(ml.model_path + cfg['ml']['encoder_model_init_file_name']))
    net_handler.models['g'].conv_decoder.load_state_dict(torch.load(ml.model_path + cfg['ml']['decoder_model_init_file_name']))

    # decoder weights only
    if cfg['ml']['adv_params']['use_decoder_weights']: return None, net_handler.models['g']

    # encoder weights only
    return net_handler.models['d'], None


  # train each label separate (used for collected net)
  if cfg['ml']['adv_params']['label_train']:

    # encoder model list
    encoder_models = torch.nn.ModuleList()
    decoder_models = torch.nn.ModuleList()

    #nn_archs = ['conv-experimental', 'adv-experimental']
    #nn_archs = ['conv-experimental']
    #nn_archs = ['adv-experimental']
    #nn_archs = ['conv-experimental', 'adv-experimental3']
    #nn_archs = ['adv-experimental3', 'conv-experimental']
    nn_archs = ['adv-experimental3']

    # number of iterations for algorithm
    num_iterations = 2

    # conv encoder for classes
    for l in cfg['datasets']['speech_commands']['sel_labels']:

      # sub model path
      sub_model_path = 'conv_encoder'

      #if check_files_existance([ml.model_path + sub_model_path + '_' + l + p for p in [cfg['ml']['encoder_model_init_file_name'], cfg['ml']['decoder_model_init_file_name']]])

      # info
      print("\nconv encoder for: ", l)
      encoder_model = None
      decoder_model = None

      # batch archive for one label
      batch_archive_l = SpeechCommandsBatchArchive(all_feature_files, batch_size=cfg['ml']['train_params']['batch_size'])


      # iterations for learning conv encoders
      for i in range(num_iterations):
      
        for j, nn_arch in enumerate(nn_archs):

          # reduce to single label
          batch_archive_l.reduce_to_label(l)

          # add noise data for conv-exp
          #if nn_arch in ['conv-experimental', 'adv-experimental3']:
          if nn_arch in ['conv-experimental']:
            #batch_archive.one_against_all(l, others_label='other', shuffle=True)
            batch_archive_l.add_noise_data(shuffle=True)

          # check classes
          print("classes: ", batch_archive_l.classes)

          # net handler
          net_handler_l = NetHandler(nn_arch=nn_arch, class_dict=batch_archive_l.class_dict, data_size=batch_archive_l.data_size, encoder_model=encoder_model, decoder_model=decoder_model, use_cpu=cfg['ml']['use_cpu'])

          # ml
          ml_l = ML(cfg_ml=cfg['ml'], audio_dataset=audio_set1, batch_archive=batch_archive_l, net_handler=net_handler_l, sub_model_path=sub_model_path, encoder_label=l)

          # change train params
          train_params = cfg['ml']['train_params'].copy()
          train_params['num_epochs'] = cfg['ml']['adv_params']['num_epochs_label']
          # if nn_arch == 'conv-experimental': train_params['num_epochs'] = 20
          # elif nn_arch == 'adv-experimental': train_params['num_epochs'] = 25
          # elif nn_arch == 'adv-experimental3': train_params['num_epochs'] = 1000

          # train and analyze
          name_ext = '_{}-{}_{}'.format(i, j, nn_arch)
          ml_l.train(new_train_params=train_params, log_on=False, name_ext=name_ext, save_models=(i==num_iterations-1))
          ml_l.analyze(name_ext=name_ext)

          # update encoder models
          if nn_arch == 'conv-experimental': encoder_model, decoder_model = net_handler_l.models['cnn'], None
          elif nn_arch in ['adv-experimental', 'adv-experimental3']: encoder_model, decoder_model = net_handler_l.models['d'], net_handler_l.models['g']

          # use decoder weights
          # elif nn_arch == 'adv-experimental3': 
          #   net_handler.models['d'].conv_encoder.transfer_decoder_weights(net_handler.models['g'].conv_decoder)
          #   encoder_model = net_handler.models['d'].conv_encoder

          #ml.analyze(name_ext=name_ext + '_after')

      # add encoder model
      encoder_models.append(encoder_model.conv_encoder)
      decoder_models.append(decoder_model.conv_decoder)

    # transfer label based conv encoders to collected conv encoder
    #collected_encoder_model.transfer_conv_weights_label_coders(encoder_models)
    #collected_decoder_model.transfer_conv_weights_label_coders(decoder_models)

    print("encoder: ", encoder_models.state_dict().keys())
    net_handler.models['d'].conv_encoder.transfer_conv_weights_label_coders(encoder_models)
    net_handler.models['g'].conv_decoder.transfer_conv_weights_label_coders(decoder_models)

    # # add noise to each weight
    # with torch.no_grad():
    #   for param in encoder_model.parameters():
    #     param.add_(torch.randn(param.shape) * 0.01)

    #torch.nn.init.xavier_uniform_(collected_encoder_model.conv_layers[1].weight, gain=torch.nn.init.calculate_gain('relu'))


  # # pre train

  # # batch archive
  # batch_archive = SpeechCommandsBatchArchive(all_feature_files, batch_size=cfg['ml']['train_params']['batch_size'])

  # # net handler for pre adv training
  # if cfg['ml']['nn_arch'] in ['conv-lim-encoder']:
  #   net_handler = NetHandler(nn_arch='adv-lim-encoder', class_dict=batch_archive.class_dict, data_size=batch_archive.data_size, encoder_model=None, decoder_model=None, use_cpu=cfg['ml']['use_cpu'])
  # else:
  #   net_handler = NetHandler(nn_arch='adv-collected-encoder', class_dict=batch_archive.class_dict, data_size=batch_archive.data_size, encoder_model=collected_encoder_model, decoder_model=collected_decoder_model, use_cpu=cfg['ml']['use_cpu'])

  # # collected encoder
  # ml = ML(cfg_ml=cfg['ml'], audio_dataset=audio_set1, batch_archive=batch_archive, net_handler=net_handler)

  # train_params = cfg['ml']['train_params'].copy()
  # train_params['num_epochs'] = 500

  # ml.analyze(name_ext='_1_pre-adv')
  # ml.train(new_train_params=train_params, log_on=False, name_ext='_adv-post_train', save_models=False)
  # ml.analyze(name_ext='_2_post-adv')

  # # update encoder model
  # collected_encoder_model = net_handler.models['d'].conv_encoder
  # collected_decoder_model = net_handler.models['g'].conv_decoder


  # # save collected model
  # torch.save(collected_encoder_model.state_dict(), ml_collected.model_path + cfg['ml']['encoder_model_init_file_name'])
  # torch.save(collected_decoder_model.state_dict(), ml_collected.model_path + cfg['ml']['decoder_model_init_file_name'])

  # # use decoder
  # if use_decoder:
  #   collected_encoder_model.transfer_decoder_weights(collected_decoder_model)

  # return collected_encoder_model

  # pre training
  if cfg['ml']['adv_params']['pre_train']:

    # set epochs
    train_params = cfg['ml']['train_params'].copy()
    #train_params['num_epochs'] = 1000
    train_params['num_epochs'] = cfg['ml']['adv_params']['num_epochs_pre']

    # training
    ml.analyze(name_ext='_1_pre-adv')
    ml.train(new_train_params=train_params, log_on=False, name_ext='_adv-post_train', save_models=True)
    ml.analyze(name_ext='_2_post-adv')

  if cfg['ml']['adv_params']['use_decoder_weights']: return None, net_handler.models['g']

  # models
  return net_handler.models['d'], None



if __name__ == '__main__':
  """
  ML - Machine Learning file
  """

  import yaml

  from batch_archive import SpeechCommandsBatchArchive
  from net_handler import NetHandler
  from audio_dataset import AudioDataset

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # audio sets
  audio_set1 = AudioDataset(cfg['datasets']['speech_commands'], cfg['feature_params'])
  audio_set2 = AudioDataset(cfg['datasets']['my_recordings'], cfg['feature_params'])

  # select feature files
  all_feature_files = audio_set1.feature_files + audio_set2.feature_files if len(audio_set1.labels) == len(audio_set2.labels) else audio_set1.feature_files

  # encoder, decoder models for certain architectures necessary
  encoder_model, decoder_model = None, None

  # conv encoder training
  if cfg['ml']['nn_arch'] in ['conv-encoder', 'conv-lim-encoder', 'conv-latent']: encoder_model, decoder_model = train_conv_encoders(cfg, audio_set1, all_feature_files)


  # create batch archive
  batch_archive = SpeechCommandsBatchArchive(all_feature_files, batch_size=cfg['ml']['train_params']['batch_size'])

  # adversarial training
  if cfg['ml']['nn_arch'] in ['adv-experimental']:
    #batch_archive.reduce_to_label('up')
    #batch_archive.add_noise_data(shuffle=True)
    batch_archive.one_against_all('up', others_label='other', shuffle=True)

  # print classes
  print("x_train: ", batch_archive.x_train.shape)

  # net handler
  net_handler = NetHandler(nn_arch=cfg['ml']['nn_arch'], class_dict=batch_archive.class_dict, data_size=batch_archive.data_size, encoder_model=encoder_model, decoder_model=decoder_model, use_cpu=cfg['ml']['use_cpu'])

  # info about models
  print("net_handler: ", net_handler.models)

  # --
  # ML

  # instance
  ml = ML(cfg_ml=cfg['ml'], audio_dataset=audio_set1, batch_archive=batch_archive, net_handler=net_handler)

  # analyze init weights
  ml.analyze(name_ext='_3_init')

  # training
  ml.train()

  # evaluation
  ml.eval()

  # analyze
  ml.analyze()



