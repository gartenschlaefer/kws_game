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

  def __init__(self, cfg_ml, audio_dataset, batch_archive, net_handler, sub_model_path=None, encoder_label=''):

    # arguments
    self.cfg_ml = cfg_ml
    self.audio_dataset = audio_dataset
    self.batch_archive = batch_archive
    self.net_handler = net_handler
    self.sub_model_path = sub_model_path
    self.encoder_label = encoder_label

    # param path ml
    self.param_path_ml = 'bs-{}_it-{}_lr-{}/'.format(self.cfg_ml['train_params']['batch_size'], self.cfg_ml['train_params']['num_epochs'], str(self.cfg_ml['train_params']['lr']).replace('.', 'p'))

    # model path
    self.model_path = self.cfg_ml['paths']['model'] + self.cfg_ml['nn_arch'] + '/' + self.audio_dataset.param_path + self.param_path_ml
    if self.sub_model_path is not None and len(self.encoder_label) is not None: self.model_path = self.model_path + sub_model_path + '_' + encoder_label + '/'

    # model file
    self.model_files = [self.model_path + model_name + '_' + self.cfg_ml['model_file_name'] for model_name, v in net_handler.models.items()]
    self.model_pre_files = [self.cfg_ml['paths']['model_pre'] + model_name + '_' + '{}_c-{}.pth'.format(self.cfg_ml['nn_arch'], self.batch_archive.n_classes) for model_name, v in net_handler.models.items()]
    
    # encoder model file
    self.encoder_model_file = None
    if self.cfg_ml['nn_arch'] in ['conv-encoder'] and len(self.encoder_label): self.encoder_model_file = self.model_path + self.cfg_ml['encoder_model_file_name']

    # params and metrics files
    self.params_file = self.model_path + self.cfg_ml['params_file_name']
    self.metrics_file = self.model_path + self.cfg_ml['metrics_file_name']
    self.info_file = self.model_path + self.cfg_ml['info_file_name']

    # image list (for adversarial)
    self.img_list = []

    # create ml folders
    create_folder(list(self.cfg_ml['paths'].values()) + [self.model_path])

    # config
    logging.basicConfig(filename=self.cfg_ml['paths']['log'] + 'ml.log', level=logging.DEBUG, format='%(asctime)s %(message)s')

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
    if check_files_existance([self.encoder_model_file] + self.model_files) and not self.cfg_ml['retrain']:

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
      self.net_handler.save_models(model_files=self.model_files, encoder_model_file=self.encoder_model_file, encoder_class_name='ConvEncoder')
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
        if k.find('conv') != -1:

          # detach weights
          v = v.detach().cpu()

          # info
          print("{} analyze: {}".format(k, v.shape))
          
          # plot images
          plot_grid_images(x=v.numpy(), padding=1, num_cols=np.clip(v.shape[0], 1, 8), title=k + self.param_path_ml.replace('/', ' ') + ' ' + self.encoder_label + name_ext, plot_path=self.model_path, name=k + name_ext, show_plot=False)
          #plot_other_grid(x=v.numpy(), title=k + self.param_path_ml.replace('/', ' '), plot_path=self.model_path, name=k, show_plot=False)

    # generate samples from trained model (only for adversarial)
    fakes = self.net_handler.generate_samples(num_samples=32, to_np=True)
    if fakes is not None:
      plot_grid_images(x=fakes, padding=1, num_cols=8, title='generated samples ' + self.encoder_label  + name_ext, plot_path=self.model_path, name='generated_samples_' +  self.encoder_label  + name_ext, show_plot=False)
      plot_mfcc_only(fakes[0, 0], fs=16000, hop=160, plot_path=self.model_path, name='generated_sample_' + self.encoder_label  + name_ext, show_plot=False)

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



def train_conv_encoders(cfg, audio_set1, audio_set2):
  """
  train convolutional encoders
  """

  import torch
  from conv_nets import ConvEncoder


  # batch archive
  batch_archive = SpeechCommandsBatchArchive(audio_set1.feature_files + audio_set2.feature_files, batch_size=cfg['ml']['train_params']['batch_size'])

  # dummy net handler
  net_handler = NetHandler(nn_arch='none', class_dict=batch_archive.class_dict, data_size=batch_archive.data_size)

  # collected encoder
  ml_collected = ML(cfg_ml=cfg['ml'], audio_dataset=audio_set1, batch_archive=batch_archive, net_handler=net_handler)

  # collected encoder model
  collected_encoder_model = ConvEncoder(batch_archive.n_classes, batch_archive.data_size, is_collection_net=True)

  # check if encoder file exists
  if check_files_existance([ml_collected.model_path + cfg['ml']['encoder_model_init_file_name']]):
    collected_encoder_model.load_state_dict(torch.load(ml_collected.model_path + cfg['ml']['encoder_model_init_file_name']))
    print("encoder model exists and loaded")

    # # add noise to each weight
    # with torch.no_grad():
    #   for param in encoder_model.parameters():
    #     param.add_(torch.randn(param.shape) * 0.01)

    #torch.nn.init.xavier_uniform_(collected_encoder_model.conv_layers[1].weight, gain=torch.nn.init.calculate_gain('relu'))

    return collected_encoder_model


  # encoder model list
  encoder_models = torch.nn.ModuleList()

  #nn_archs = ['conv-experimental', 'adv-experimental']
  #nn_archs = ['conv-experimental']
  #nn_archs = ['adv-experimental']
  #nn_archs = ['conv-experimental', 'adv-experimental3']
  #nn_archs = ['adv-experimental3', 'conv-experimental']
  nn_archs = ['adv-experimental3']

  # number of iterations for algorithm
  num_iterations = 1

  # conv encoder for classes
  for l in cfg['datasets']['speech_commands']['sel_labels']:

    # info
    print("\nconv encoder for: ", l)
    encoder_model = None

    # batch archive for one label
    batch_archive = SpeechCommandsBatchArchive(audio_set1.feature_files + audio_set2.feature_files, batch_size=cfg['ml']['train_params']['batch_size'])


    # iterations for learning conv encoders
    for i in range(num_iterations):
    
      for j, nn_arch in enumerate(nn_archs):

        # reduce to single label
        batch_archive.reduce_to_label(l)

        # add noise data for conv-exp
        #if nn_arch in ['conv-experimental', 'adv-experimental3']:
        if nn_arch in ['conv-experimental']:
          #batch_archive.one_against_all(l, others_label='other', shuffle=True)
          batch_archive.add_noise_data(shuffle=True)

        # check classes
        print("classes: ", batch_archive.classes)

        # net handler
        net_handler = NetHandler(nn_arch=nn_arch, class_dict=batch_archive.class_dict, data_size=batch_archive.data_size, encoder_model=encoder_model, use_cpu=cfg['ml']['use_cpu'])

        # ml
        ml = ML(cfg_ml=cfg['ml'], audio_dataset=audio_set1, batch_archive=batch_archive, net_handler=net_handler, sub_model_path='conv_encoder', encoder_label=l)

        # change train params
        train_params = cfg['ml']['train_params'].copy()
        if nn_arch == 'conv-experimental': train_params['num_epochs'] = 20
        elif nn_arch == 'adv-experimental': train_params['num_epochs'] = 25
        elif nn_arch == 'adv-experimental3': train_params['num_epochs'] = 5000

        # train and analyze
        name_ext = '_{}-{}_{}'.format(i, j, nn_arch)
        ml.train(new_train_params=train_params, log_on=False, name_ext=name_ext, save_models=(i==num_iterations-1))
        ml.analyze(name_ext=name_ext)

        # update encoder models
        if nn_arch == 'conv-experimental': encoder_model = net_handler.models['cnn'].conv_encoder
        elif nn_arch == 'adv-experimental': encoder_model = net_handler.models['d'].conv_encoder
        #elif nn_arch == 'adv-experimental3': encoder_model = net_handler.models['d'].conv_encoder

        # use decoder weights
        elif nn_arch == 'adv-experimental3': 
          net_handler.models['d'].conv_encoder.transfer_decoder_weights(net_handler.models['g'].conv_decoder)
          encoder_model = net_handler.models['d'].conv_encoder

        #ml.analyze(name_ext=name_ext + '_after')

    # add encoder model
    encoder_models.append(encoder_model)

  # transfer label based conv encoders to collected conv encoder
  collected_encoder_model.transfer_conv_weights(encoder_models)


  # # add noise to each weight
  # with torch.no_grad():
  #   for param in encoder_model.parameters():
  #     param.add_(torch.randn(param.shape) * 0.01)

  #torch.nn.init.xavier_uniform_(collected_encoder_model.conv_layers[1].weight, gain=torch.nn.init.calculate_gain('relu'))


  # # pre train

  # # batch archive
  # batch_archive = SpeechCommandsBatchArchive(audio_set1.feature_files + audio_set2.feature_files, batch_size=cfg['ml']['train_params']['batch_size'])

  # # dummy net handler
  # net_handler = NetHandler(nn_arch='adv-collected-encoder', class_dict=batch_archive.class_dict, data_size=batch_archive.data_size, encoder_model=collected_encoder_model, use_cpu=cfg['ml']['use_cpu'])

  # # collected encoder
  # ml = ML(cfg_ml=cfg['ml'], audio_dataset=audio_set1, batch_archive=batch_archive, net_handler=net_handler)

  # ml.analyze(name_ext='adv-pre')
  # ml.train(new_train_params=train_params, log_on=False, name_ext='adv-post', save_models=False)

  # collected_encoder_model = net_handler.models['d'].conv_encoder


  # save collected model
  torch.save(collected_encoder_model.state_dict(), ml_collected.model_path + cfg['ml']['encoder_model_init_file_name'])

  return collected_encoder_model



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

  # encoder models for certain architectures necessary
  encoder_model = None

  # conv encoder training
  if cfg['ml']['nn_arch'] in ['conv-encoder']: encoder_model = train_conv_encoders(cfg, audio_set1, audio_set2)


  # create batch archive
  batch_archive = SpeechCommandsBatchArchive(audio_set1.feature_files + audio_set2.feature_files, batch_size=cfg['ml']['train_params']['batch_size'])

  # adversarial training
  if cfg['ml']['nn_arch'] in ['adv-experimental']:
    #batch_archive.reduce_to_label('up')
    #batch_archive.add_noise_data(shuffle=True)
    batch_archive.one_against_all('up', others_label='other', shuffle=True)

  # print classes
  print("x_train: ", batch_archive.x_train.shape)

  # net handler
  net_handler = NetHandler(nn_arch=cfg['ml']['nn_arch'], class_dict=batch_archive.class_dict, data_size=batch_archive.data_size, encoder_model=encoder_model, use_cpu=cfg['ml']['use_cpu'])

  # info about models
  print("net_handler: ", net_handler.models)

  # --
  # ML

  # instance
  ml = ML(cfg_ml=cfg['ml'], audio_dataset=audio_set1, batch_archive=batch_archive, net_handler=net_handler)

  # analyze init weights
  ml.analyze(name_ext='_init')

  # training
  ml.train()

  # evaluation
  ml.eval()

  # analyze
  ml.analyze()



