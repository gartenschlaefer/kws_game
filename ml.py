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

  def __init__(self, cfg_ml, audio_dataset, batch_archive, net_handler, sub_model_path=None, encoder_label=None):

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
    if self.sub_model_path is not None and self.encoder_label is not None: self.model_path = self.model_path + sub_model_path + '_' + encoder_label + '/'

    # model file
    self.model_files = [self.model_path + model_name + '_' + self.cfg_ml['model_file_name'] for model_name, v in net_handler.models.items()]
    self.model_pre_files = [self.cfg_ml['paths']['model_pre'] + model_name + '_' + '{}_c-{}.pth'.format(self.cfg_ml['nn_arch'], self.batch_archive.n_classes) for model_name, v in net_handler.models.items()]
    
    # encoder model file
    self.encoder_model_file = None
    if self.cfg_ml['nn_arch'] in ['conv-encoder']: self.encoder_model_file = self.model_path + self.cfg_ml['encoder_model_file_name']

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

    # load pre trained model
    if self.cfg_ml['load_pre_model']:
      self.net_handler.load_models(model_files=self.model_pre_files)


  def train(self):
    """
    training
    """

    # check if model already exists
    if check_files_existance([self.encoder_model_file] + self.model_files) and not self.cfg_ml['retrain']:

      # load model and check if it was possible
      if self.net_handler.load_models(model_files=self.model_files): return

    # train
    train_score = self.net_handler.train_nn(train_params=self.cfg_ml['train_params'], batch_archive=self.batch_archive, callback_f=self.image_collect)

    # training info
    logging.info('Traning on arch: [{}], train_params: {}, device: [{}], time: {}'.format(self.cfg_ml['nn_arch'], self.cfg_ml['train_params'], self.net_handler.device, s_to_hms_str(train_score.time_usage)))
    
    # save training results
    self.net_handler.save_models(model_files=self.model_files, encoder_model_file=self.encoder_model_file, encoder_class_name='ConvEncoder')

    # save params, metrics and infos
    self.save_params()
    self.save_metrics(train_score=train_score)
    self.save_infos()

    # save as pre trained model
    if self.cfg_ml['save_as_pre_model']:
      self.net_handler.save_models(model_files=self.model_pre_files)

    # plot score
    plot_train_score(train_score, plot_path=self.model_path)


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

    # activate eval mode (no dropout layers)
    self.net_handler.set_eval_mode()

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


  def analyze(self):
    """
    analyze function, e.g. analyze weights
    """

    # analyze weights
    weights = self.net_handler.get_model_weights()

    if weights is not None:

      # go through each weight
      for k, v in weights.items():

        # convolutional layers
        if k.startswith('conv'):

          # info
          print("{} analyze: {}".format(k, v.shape))
          
          # plot images
          plot_grid_images(x=v.numpy(), padding=1, num_cols=8, title=k + self.param_path_ml.replace('/', ' '), plot_path=self.model_path, name=k, show_plot=False)
          #plot_other_grid(x=v.numpy(), title=k + self.param_path_ml.replace('/', ' '), plot_path=self.model_path, name=k, show_plot=False)

    # generate samples (for generative networks)
    self.generate_samples()

    # animation (for generative networks)
    if self.cfg_ml['plot_animation']: self.create_anim()


  def generate_samples(self):
    """
    generate samples if it is a generative network
    """

    # generate samples from trained model
    fake = self.net_handler.generate_samples(num_samples=1, to_np=True)
    if fake is not None:
      plot_mfcc_only(fake, fs=16000, hop=160, plot_path=self.model_path, name='generated_sample', show_plot=False)


  def image_collect(self, x):
    """
    collect images of mfcc's (used as callback function in the training of adversarial networks)
    """

    # append image
    self.img_list.append(x)


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

  if not cfg['ml']['nn_arch'].startswith('adv'):
    print("***no adversarial net used for conv encoders")

  import torch

  # encoder model list
  encoder_models = torch.nn.ModuleList()

  # conv encoder for classes
  for l in cfg['datasets']['speech_commands']['sel_labels']:
    
    # info
    print("\nconv encoder for: ", l)

    # batch archive for one label
    batch_archive = SpeechCommandsBatchArchive(audio_set1.feature_files + audio_set2.feature_files, batch_size=cfg['ml']['train_params']['batch_size'])
    
    #batch_archive.reduce_to_label(l)
    batch_archive.one_against_all(l, others_label='other', shuffle=True)
    print("classes: ", batch_archive.classes)

    # net handler
    net_handler = NetHandler(nn_arch='adv-experimental', n_classes=batch_archive.n_classes, data_size=batch_archive.data_size, use_cpu=cfg['ml']['use_cpu'])
    #net_handler = NetHandler(nn_arch='conv-experimental', n_classes=batch_archive.n_classes, data_size=batch_archive.data_size, use_cpu=cfg['ml']['use_cpu'])

    # ml
    ml = ML(cfg_ml=cfg['ml'], audio_dataset=audio_set1, batch_archive=batch_archive, net_handler=net_handler, sub_model_path='conv_encoder', encoder_label=l)
    
    # train and analyze
    ml.train()
    ml.analyze()

    # add encoder model
    encoder_models.append(net_handler.models['d'].conv_encoder)
    #encoder_models.append(net_handler.models['cnn'])

  return encoder_models



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
  encoder_models = None

  # conv encoder training
  if cfg['ml']['nn_arch'] in ['conv-encoder']:
    encoder_models = train_conv_encoders(cfg, audio_set1, audio_set2)


  # create batch archive
  batch_archive = SpeechCommandsBatchArchive(audio_set1.feature_files + audio_set2.feature_files, batch_size=cfg['ml']['train_params']['batch_size'])

  #if cfg['ml']['nn_arch'] in ['adv-experimental']:
  #  batch_archive.reduce_to_label('up')
  #batch_archive.reduce_to_label('up')
  #batch_archive.add_noise_data(shuffle=True)
  #batch_archive.one_against_all('down', others_label='other', shuffle=True)

  # print classes
  print("x_train: ", batch_archive.x_train.shape)

  # net handler
  net_handler = NetHandler(nn_arch=cfg['ml']['nn_arch'], n_classes=batch_archive.n_classes, data_size=batch_archive.data_size, encoder_models=encoder_models, use_cpu=cfg['ml']['use_cpu'])

  # info about models
  print("net_handler: ", net_handler.models)

  # --
  # ML

  # instance
  ml = ML(cfg_ml=cfg['ml'], audio_dataset=audio_set1, batch_archive=batch_archive, net_handler=net_handler)

  # training
  ml.train()

  # evaluation
  ml.eval()

  # analyze
  ml.analyze()



