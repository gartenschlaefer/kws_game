"""
Handling Neural Networks
"""

import numpy as np
import torch
import time

from conv_nets import ConvNetTrad, ConvNetFstride4, ConvNetExperimental
from adversarial_nets import G_experimental, D_experimental
from score import TrainScore, EvalScore


class NetHandler():
  """
  Neural Network Handler with general functionalities and interfaces
  """

  def __new__(cls, nn_arch, n_classes, data_size, use_cpu=False):

    # adversarial handler
    if nn_arch == 'adv-experimental':
      for child_cls in cls.__subclasses__():
        if child_cls.__name__ == 'AdversarialNetHandler':
          return super().__new__(AdversarialNetHandler)

    # cnn handler
    elif nn_arch == 'conv-trad' or nn_arch == 'conv-fstride' or nn_arch == 'conv-experimental':
      for child_cls in cls.__subclasses__():
        if child_cls.__name__ == 'CnnHandler':
          return super().__new__(CnnHandler)

    # handler specific
    return super().__new__(cls)


  def __init__(self, nn_arch, n_classes, data_size, use_cpu=False):

    # arguments
    self.nn_arch = nn_arch
    self.n_classes = n_classes
    self.data_size = data_size
    self.use_cpu = use_cpu

    # vars
    self.num_print_per_epoch = 2

    # set device
    self.device = torch.device("cuda:0" if (torch.cuda.is_available() and not self.use_cpu) else "cpu")

    # print msg
    print("device: ", self.device)
    if torch.cuda.is_available() and not self.use_cpu: print("use gpu: ", torch.cuda.get_device_name(self.device))

    # models dictionary key: name, value: model
    self.models = {}


  def init_models(self):
    """
    instantiate the requested models and sent them to the device
    """

    # select network architecture
    if self.nn_arch == 'conv-trad': self.models = {'cnn':ConvNetTrad(self.n_classes, self.data_size)}
    elif self.nn_arch == 'conv-fstride': self.models = {'cnn':ConvNetFstride4(self.n_classes, self.data_size)}
    elif self.nn_arch == 'conv-experimental': self.models = {'cnn':ConvNetExperimental(self.n_classes, self.data_size)}
    elif self.nn_arch == 'adv-experimental': self.models = {'g':G_experimental(), 'd':D_experimental()}
    else: print("***Network Architecture not found!")

    # send models to device
    self.models = dict((k, model.to(self.device)) for k, model in self.models.items())


  def set_eval_mode(self):
    """
    sets the eval mode (dropout layers are ignored)
    """
    self.models = dict((k, model.eval()) for k, model in self.models.items())


  def set_train_mode(self):
    """
    sets the eval mode (dropout layers are ignored)
    """
    self.models = dict((k, model.train()) for k, model in self.models.items())


  def print_train_info(self, epoch, mini_batch, train_score, k_print=10):
    """
    print some training info
    """

    do_print_anyway = False

    if not k_print:
      k_print = 1
      do_print_anyway = True

    # print loss
    if mini_batch % k_print == k_print-1 or do_print_anyway:

      # adversarial gets separate print
      if train_score.use_adv:
        print('epoch: {}, mini-batch: {}, G loss: [{:.5f}], D loss real: [{:.5f}], D loss fake: [{:.5f}]'.format(epoch + 1, mini_batch + 1, train_score.g_batch_loss / k_print, train_score.d_batch_loss_real / k_print, train_score.d_batch_loss_fake / k_print))

      else:
        # print info
        print('epoch: {}, mini-batch: {}, loss: [{:.5f}]'.format(epoch + 1, mini_batch + 1, train_score.batch_loss / k_print))


  def load_models(self, model_files):
    """
    loads model from array of model file names
    watch order if more than one model file
    """

    # safety check
    if len(model_files) != len(self.models):
      print("***len of model file names is not equal length of models")
      return

    # load models
    for model_file, (k, model) in zip(model_files, self.models.items()):

      try:
        print("load model: {}, net handler model: {}".format(model_file, k))
        model.load_state_dict(torch.load(model_file))

      except:
        print("\n***could not load model!!!\n")   
      pass


  def save_models(self, model_files):
    """
    saves model
    """

    # safety check
    if len(model_files) != len(self.models):
      print("***len of model file names is not equal length of models")
      return

    # load models
    for model_file, (k, model) in zip(model_files, self.models.items()):

      try:
        print("save model: {}, net handler model: {}".format(model_file, k))
        torch.save(model.state_dict(), model_file)

      except:
        print("\n***could not save model!!!\n")


  def save_params(self, params_file, train_params, class_dict):
    """
    save parameter file
    """
    np.savez(params_file, nn_arch=self.nn_arch, train_params=train_params, class_dict=class_dict, data_size=self.data_size)


  def save_metrics(self, metrics_file, train_score=None):
    """
    save training metrics
    """
    np.savez(metrics_file, train_score=train_score)


  def train_nn(self, train_params, batch_archive, callback_f=None):
    """
    train interface
    """
    return TrainScore(train_params['num_epochs'])


  def eval_nn(self, eval_set, batch_archive, calc_cm=False, verbose=False):
    """
    evaluation interface
    """
    return EvalScore()


  def eval_select_set(self, eval_set, batch_archive):
    """
    select set to evaluate (only for batch archive class)
    """

    # select the set
    x_eval, y_eval, z_eval = None, None, None

    # validation set
    if eval_set == 'val':
      x_eval, y_eval, z_eval = batch_archive.x_val, batch_archive.y_val, batch_archive.z_val

    # test set
    elif eval_set == 'test':
      x_eval, y_eval, z_eval = batch_archive.x_test, batch_archive.y_test, batch_archive.z_test

    # my test set
    elif eval_set == 'my':
      x_eval, y_eval, z_eval = batch_archive.x_my, batch_archive.y_my, batch_archive.z_my

    # set not found
    else:
      print("wrong usage of eval_nn, select eval_set one out of ['val', 'test', 'my']")

    return x_eval, y_eval, z_eval


  def classify_sample(self, x):
    """
    classify a single sample
    """
    y_hat, o = 0, 0
    return y_hat, o


  def generate_samples(self, noise=None, num_samples=10, to_np=False):
    """
    generate samples if it is a generative network
    """
    return None


  def set_eval_mode(self):
    """
    sets the eval mode (dropout layers are ignored)
    """
    pass



class CnnHandler(NetHandler):
  """
  Neural Network Handler for CNNs
  """

  def __init__(self, nn_arch, n_classes, data_size, use_cpu=False):

    # parent class init
    super().__init__(nn_arch, n_classes, data_size, use_cpu)

    # loss criterion
    self.criterion = torch.nn.CrossEntropyLoss()

    # init models
    self.init_models()


  def train_nn(self, train_params, batch_archive, callback_f=None):
    """
    train the neural network
    train_params: {'num_epochs': [], 'lr': [], 'momentum': []}
    """

    # create optimizer
    #optimizer = torch.optim.SGD(self.model.parameters(), lr=train_params['lr'], momentum=train_params['momentum'])
    optimizer = torch.optim.Adam(self.models['cnn'].parameters(), lr=train_params['lr'])

    # score collector
    train_score = TrainScore(train_params['num_epochs'])

    print("\n--Training starts:")

    # start time
    start_time = time.time()

    # epochs
    for epoch in range(train_params['num_epochs']):

      # TODO: do this with loader function from pytorch (maybe or not)
      # fetch data samples
      for i, (x, y) in enumerate(zip(batch_archive.x_train.to(self.device), batch_archive.y_train.to(self.device))):

        # zero parameter gradients
        optimizer.zero_grad()

        # forward pass o:[b x c]
        o = self.models['cnn'](x)

        # loss
        loss = self.criterion(o, y)

        # backward
        loss.backward()

        # optimizer step - update params
        optimizer.step()

        # update batch loss collection
        train_score.update_batch_losses(epoch, loss.item())

        # print some infos
        self.print_train_info(epoch, i, train_score, k_print=batch_archive.y_train.shape[0] // self.num_print_per_epoch)
        train_score.reset_batch_losses()

      # valdiation
      eval_score = self.eval_nn('val', batch_archive)

      # update score collector
      train_score.val_loss[epoch], train_score.val_acc[epoch] = eval_score.loss, eval_score.acc

    # TODO: Early stopping if necessary

    print('--Training finished')

    # log time
    train_score.time_usage = time.time() - start_time 

    return train_score


  def eval_nn(self, eval_set, batch_archive, calc_cm=False, verbose=False):
    """
    evaluation of nn
    use eval_set out of ['val', 'test', 'my']
    """

    # select the evaluation set
    x_eval, y_eval, z_eval = self.eval_select_set(eval_set, batch_archive)

    # if set does not exist
    if x_eval is None or y_eval is None:
      print("no eval set found")
      return EvalScore(calc_cm=calc_cm)

    # init score
    eval_score = EvalScore(calc_cm=calc_cm)


    # no gradients for eval
    with torch.no_grad():

      # load data
      for i, (x, y) in enumerate(zip(x_eval.to(self.device), y_eval.to(self.device))):

        # classify
        o = self.models['cnn'](x)

        # loss
        loss = self.criterion(o, y)

        # prediction
        _, y_hat = torch.max(o.data, 1)

        # update eval score
        eval_score.update(loss, y.cpu(), y_hat.cpu())

        # some prints
        if verbose:
          if z_eval is not None:
            print("\nlabels: {}".format(z_eval[i]))
          print("output: {}\npred: {}\nactu: {}, \t corr: {} ".format(o.data, y_hat, y, (y_hat == y).sum().item()))

    # finish up scores
    eval_score.finish()

    return eval_score


  def classify_sample(self, x):
    """
    classification of a single sample presented in dim [m x f]
    """

    # input to tensor
    x = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(x.astype(np.float32)), 0), 0).to(self.device)

    # no gradients for eval
    with torch.no_grad():

      # classify
      o = self.models['cnn'](x)

      # prediction
      _, y_hat = torch.max(o.data, 1)

    return int(y_hat), o


  def get_model_weights(self):
    """
    get model weights
    """
    return self.models['cnn'].get_weights()



class AdversarialNetHandler(NetHandler):
  """
  Adversarial Neural Network Handler
  adapted form: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
  """

  def __init__(self, nn_arch, n_classes, data_size, use_cpu=False):

    # parent class init
    super().__init__(nn_arch, n_classes, data_size, use_cpu)

    # loss criterion
    self.criterion = torch.nn.BCELoss()

    # neural network models, G-Generator, D-Discriminator
    self.init_models()

    # labels
    self.real_label = 1.
    self.fake_label = 0.

    # weights init
    self.models['g'].apply(self.weights_init)
    self.models['d'].apply(self.weights_init)


  def weights_init(self, m):
    """
    custom weights initialization called on netG and netD
    adapted form: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    """

    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
      torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
      torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
      torch.nn.init.constant_(m.bias.data, 0)


  def train_nn(self, train_params, batch_archive, callback_f=None):
    """
    train adversarial nets
    """

    # Create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(32, self.models['g'].n_latent, device=self.device)

    # Setup Adam optimizers for both G and D
    optimizer_d = torch.optim.Adam(self.models['d'].parameters(), lr=train_params['lr'], betas=(train_params['beta'], 0.999))
    optimizer_g = torch.optim.Adam(self.models['g'].parameters(), lr=train_params['lr'], betas=(train_params['beta'], 0.999))

    # score collector
    train_score = TrainScore(train_params['num_epochs'], use_adv=True)

    print("\n--Training starts:")

    # start time
    start_time = time.time()

    # epochs
    for epoch in range(train_params['num_epochs']):

      # fetch data samples
      for i, x in enumerate(batch_archive.x_train.to(self.device)):

        # zero parameter gradients
        self.models['d'].zero_grad()
        optimizer_d.zero_grad()

        # --
        # train with real batch

        # labels for batch
        y = torch.full((batch_archive.batch_size,), self.real_label, dtype=torch.float, device=self.device)

        # forward pass o:[b x c]
        o = self.models['d'](x).view(-1)

        # loss of D with reals
        d_loss_real = self.criterion(o, y)
        d_loss_real.backward()
        #d_reals = o.mean().item()

        # --
        # train with fake batch

        # create noise as input
        noise = torch.randn(batch_archive.batch_size, self.models['g'].n_latent, device=self.device)

        # create fakes through Generator
        fakes = self.models['g'](noise)

        # create fake labels
        y.fill_(self.fake_label)

        # fakes to D
        o = self.models['d'](fakes.detach()).view(-1)

        # loss of D with fakes
        d_loss_fake = self.criterion(o, y)
        d_loss_fake.backward()
        #D_G_z1 = o.mean().item()

        # cumulate all losses
        #d_loss = loss_real + lossD_fake

        # optimizer step
        optimizer_d.step()

        # --
        # update of G

        self.models['g'].zero_grad()

        y.fill_(self.real_label)

        o = self.models['d'](fakes).view(-1)

        # loss of G of D with fakes
        g_loss = self.criterion(o, y)
        g_loss.backward()
        #D_G_z2 = o.mean().item()

        # optimizer step
        optimizer_g.step()

        # update batch loss collection
        train_score.update_batch_losses(epoch, loss=0.0, g_loss=g_loss.item(), d_loss_real=d_loss_real.item(), d_loss_fake=d_loss_fake.item())

        # print some infos
        self.print_train_info(epoch, i, train_score, k_print=batch_archive.y_train.shape[0] // self.num_print_per_epoch)
        train_score.reset_batch_losses()

      # check progess after epoch with callback function
      if callback_f is not None:
        callback_f(self.generate_samples(noise=fixed_noise, to_np=True))

    print('--Training finished')

    # log time
    train_score.time_usage = time.time() - start_time 

    return train_score


  def generate_samples(self, noise=None, num_samples=10, to_np=False):
    """
    generator samples from G
    """

    # generate noise if not given
    if noise is None:
      noise = torch.randn(num_samples, self.models['g'].n_latent, device=self.device)

    # create fakes through Generator
    with torch.no_grad():
      fakes = torch.squeeze(self.models['g'](noise).detach().cpu())

    # to numpy if necessary
    if to_np:
      fakes = fakes.numpy()

    return fakes



if __name__ == '__main__':
  """
  handles all neural networks with training, evaluation and classifying samples 
  """

  import yaml

  from batch_archive import SpeechCommandsBatchArchive
  from audio_dataset import AudioDataset

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # audio sets
  audio_set1 = AudioDataset(cfg['datasets']['speech_commands'], cfg['feature_params'])
  audio_set2 = AudioDataset(cfg['datasets']['my_recordings'], cfg['feature_params'])

  # create batches
  batch_archive = SpeechCommandsBatchArchive(audio_set1.feature_files + audio_set2.feature_files, batch_size=32, batch_size_eval=4)
  print("data: ", batch_archive.data_size)

  # choose architecture
  nn_arch = 'conv-fstride'
  #nn_arch = 'conv-trad'
  #nn_arch = 'adv-experimental'

  # create an cnn handler
  net_handler = NetHandler(nn_arch=cfg['ml']['nn_arch'], n_classes=batch_archive.n_classes, data_size=batch_archive.data_size, use_cpu=cfg['ml']['use_cpu'])
  print(net_handler.models)

  # training
  net_handler.train_nn(cfg['ml']['train_params'], batch_archive=batch_archive)

  # validation
  net_handler.eval_nn(eval_set='val', batch_archive=batch_archive, calc_cm=False, verbose=False)

  # classify sample
  y_hat, o = net_handler.classify_sample(np.random.randn(net_handler.data_size[1], net_handler.data_size[2]))

  print("classify: [{}]\noutput: [{}]".format(y_hat, o))

  # analyze model weights
  net_handler.analyze_model_weights()

