"""
Neural Network Handling
"""

import numpy as np
import torch
import time
import sys
import re

# nets
from conv_nets import ConvNetTrad, ConvNetFstride4, ConvNetExperimental, ConvJim
from adversarial_nets import Adv_G_Experimental, Adv_D_Experimental, Adv_G_Jim, Adv_D_Jim
from hybrid_nets import HybJim
from wavenet import Wavenet

# other
from score import TrainScore, AdversarialTrainScore, HybridTrainScore, WavenetTrainScore, EvalScore


class NetHandler():
  """
  Neural Network Handler with general functionalities and interfaces
  """

  def __new__(cls, nn_arch, class_dict, data_size, feature_params, encoder_model=None, decoder_model=None, use_cpu=False):

    # adversarial handler
    if nn_arch in ['adv-experimental']:
      for child_cls in cls.__subclasses__():
        if child_cls.__name__ == 'AdversarialNetHandler':
          return super().__new__(AdversarialNetHandler)

    # adversarial sim handler
    if nn_arch in ['adv-jim', 'adv-jim-label']:
      for child_cls in cls.__subclasses__():
        for child_child_cls in child_cls.__subclasses__():
          if child_child_cls.__name__ == 'AdversarialSimNetHandler':
            return super().__new__(AdversarialSimNetHandler)

    # cnn handler
    elif nn_arch in ['conv-experimental', 'conv-trad', 'conv-fstride', 'conv-jim']:
      for child_cls in cls.__subclasses__():
        if child_cls.__name__ == 'CnnHandler':
          return super().__new__(CnnHandler)

    # hybrid handler
    elif nn_arch in ['hyb-jim']:
      for child_cls in cls.__subclasses__():
        if child_cls.__name__ == 'HybridNetHandler':
          return super().__new__(HybridNetHandler)

    # wavenet handler
    elif nn_arch in ['wavenet']:
      for child_cls in cls.__subclasses__():
        if child_cls.__name__ == 'WavenetHandler':
          return super().__new__(WavenetHandler)

    # handler specific
    return super().__new__(cls)


  def __init__(self, nn_arch, class_dict, data_size, feature_params, encoder_model=None, decoder_model=None, use_cpu=False):

    # arguments
    self.nn_arch = nn_arch
    self.class_dict = class_dict
    self.data_size = data_size
    self.feature_params = feature_params
    self.encoder_model = encoder_model
    self.decoder_model = decoder_model
    self.use_cpu = use_cpu

    # vars
    self.n_classes = len(self.class_dict)
    self.num_print_per_epoch = 2

    # set device
    self.device = torch.device("cuda:0" if (torch.cuda.is_available() and not self.use_cpu) else "cpu")

    # print device
    print("net handler device: {}\ngpu: {}".format(self.device, torch.cuda.get_device_name(self.device))) if torch.cuda.is_available() else print("net handler device: {}".format(self.device))

    # models dictionary key: name, value: model
    self.models = {}


  def init_models(self):
    """
    instantiate the requested models and sent them to the device
    """

    # cnns
    if self.nn_arch == 'conv-trad': self.models = {'cnn': ConvNetTrad(self.n_classes, self.data_size)}
    elif self.nn_arch == 'conv-fstride': self.models = {'cnn': ConvNetFstride4(self.n_classes, self.data_size)}
    elif self.nn_arch == 'conv-experimental': self.models = {'cnn': ConvNetExperimental(self.n_classes, self.data_size)}
    elif self.nn_arch == 'conv-jim': self.models = {'cnn': ConvJim(self.n_classes, self.data_size)}

    # adversarials
    elif self.nn_arch == 'adv-experimental': self.models = {'g': Adv_G_Experimental(self.n_classes, self.data_size, is_last_activation_sigmoid=self.feature_params['norm_features']), 'd': Adv_D_Experimental(self.n_classes, self.data_size)}
    elif self.nn_arch == 'adv-jim': self.models = {'g': Adv_G_Jim(self.n_classes, self.data_size, is_last_activation_sigmoid=self.feature_params['norm_features']), 'd': Adv_D_Jim(self.n_classes, self.data_size)}
    elif self.nn_arch == 'adv-jim-label': self.models = {'g': Adv_G_Jim(self.n_classes, self.data_size, n_feature_maps_l0=8, is_last_activation_sigmoid=self.feature_params['norm_features']), 'd': Adv_D_Jim(self.n_classes, self.data_size, n_feature_maps_l0=8)}
  
    # hybrid
    elif self.nn_arch == 'hyb-jim': self.models = {'hyb': HybJim(self.n_classes, self.data_size), 'g': Adv_G_Jim(self.n_classes, self.data_size, is_last_activation_sigmoid=self.feature_params['norm_features'])}

    # wavenet
    elif self.nn_arch == 'wavenet': self.models = {'wav': Wavenet(self.n_classes)}

    # not found
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
    sets the training mode (dropout layers active)
    """
    self.models = dict((k, model.train()) for k, model in self.models.items())


  def load_models(self, model_files):
    """
    loads models
    """

    # safety check
    if len(model_files) != len(self.models): print("***load models failed: len of model file names is not equal length of models"), sys.exit()

    # model names from files
    f_model_name_dict = {re.sub(r'_model\.pth', '', re.findall(r'[\w]+_model\.pth', mf)[0]): mf for mf in model_files}

    # load models
    [[(print("\nload model: {}\nnet handler model: {}".format(f_model, model_name)), model.load_state_dict(torch.load(f_model))) for f_model_name, f_model in f_model_name_dict.items() if f_model_name == model_name] for model_name, model in self.models.items()]


  def save_models(self, model_files):
    """
    saves models
    """

    # safety check
    if len(model_files) != len(self.models): print("***save models failed: len of model file names is not equal length of models"), sys.exit()

    # model names from files
    f_model_name_dict = {re.sub(r'_model\.pth', '', re.findall(r'[\w]+_model\.pth', mf)[0]): mf for mf in model_files}

    # save models
    [[(print("\nsave model: {}\nnet handler model: {}".format(f_model, model_name)), torch.save(model.state_dict(), f_model)) for f_model_name, f_model in f_model_name_dict.items() if f_model_name == model_name] for model_name, model in self.models.items()]


  def set_up_training(self, train_params):
    """
    setup training
    """
    pass


  def update_training_params(self, epoch, train_params):
    """
    update training parameters upon epoch
    """
    pass


  def train_nn(self, train_params, batch_archive, callback_f=None):
    """
    train interface
    """
    self.set_up_training(train_params)
    return TrainScore(train_params['num_epochs'])


  def eval_nn(self, eval_set_name, batch_archive, collect_things=False, verbose=False):
    """
    evaluation of nn
    use eval_set out of ['val', 'test', 'my']
    """

    # eval mode
    self.set_eval_mode()

    # check if set name is in batch archive
    if not eval_set_name in batch_archive.eval_set_names: print("***evaluation set is not in batch archive"), sys.exit()

    # if set does not exist
    if batch_archive.x_batch_dict[eval_set_name] is None or batch_archive.y_batch_dict[eval_set_name] is None:
      print("no eval set found")
      return EvalScore(eval_set_name=eval_set_name, class_dict=self.class_dict, collect_things=collect_things)

    # init score
    eval_score = EvalScore(eval_set_name=eval_set_name, class_dict=self.class_dict, collect_things=collect_things)

    # no gradients for eval
    with torch.no_grad():

      # load data
      for mini_batch, (x, y, z) in enumerate(zip(batch_archive.x_batch_dict[eval_set_name].to(self.device), batch_archive.y_batch_dict[eval_set_name].to(self.device), batch_archive.z_batch_dict[eval_set_name])):

        # forward pass
        eval_score = self.eval_forward(mini_batch, x, y, z, eval_score, verbose=verbose)

    # finish up scores
    eval_score.finish()

    # train mode
    self.set_train_mode()

    return eval_score


  def eval_forward(self, mini_batch, x, y, z, eval_score, verbose=False):
    """
    eval forward pass
    """
    return eval_score


  def classify_sample(self, x):
    """
    classify a single sample
    """
    y_hat, o, label = 0, 0.0, 'nothing'
    return y_hat, o, label


  def generate_samples(self, noise=None, num_samples=10, to_np=False):
    """
    generate samples if it is a generative network
    """
    return None


  def count_params_and_mults(self):
    """
    count networks parameters and multiplications
    """

    # count dictionary
    count_dict = {}

    # go through each model
    for k, model in self.models.items(): count_dict.update({k + '_params_layers': [p.numel() for p in model.parameters() if p.requires_grad]})

    return count_dict



class CnnHandler(NetHandler):
  """
  Neural Network Handler for CNNs
  """

  def __init__(self, nn_arch, class_dict, data_size, feature_params, encoder_model=None, decoder_model=None, use_cpu=False):

    # parent class init
    super().__init__(nn_arch, class_dict, data_size, feature_params, encoder_model=encoder_model, decoder_model=decoder_model, use_cpu=use_cpu)

    # loss criterion
    self.criterion = torch.nn.CrossEntropyLoss()

    # init models
    self.init_models()

    # transfer params safety
    if bool(self.encoder_model is not None) and bool(self.decoder_model is not None): print("***used both decoder and encoder weights"), sys.exit()

    # transfer params
    if self.encoder_model is not None: self.models['cnn'].transfer_params(model=self.encoder_model)
    elif self.decoder_model is not None: self.models['cnn'].transfer_params(model=self.decoder_model)


  def set_up_training(self, train_params):
    """
    set optimizer in training
    """

    # create optimizer
    self.optimizer = torch.optim.Adam(self.models['cnn'].parameters(), lr=train_params['lr'], betas=(train_params['beta'], 0.999))


  def train_nn(self, train_params, batch_archive, callback_f=None):
    """
    training of the neural network train_params: {'num_epochs': [], 'lr': [], 'momentum': []}
    """

    # setup training
    self.set_up_training(train_params)

    # score collector
    train_score = TrainScore(train_params['num_epochs'], invoker_class_name=self.__class__.__name__, k_print=batch_archive.y_batch_dict['train'].shape[0] // self.num_print_per_epoch)

    # epochs
    for epoch in range(train_params['num_epochs']):

      # update training params if necessary
      self.update_training_params(epoch, train_params)

      # fetch data samples
      for mini_batch, (x, y) in enumerate(zip(batch_archive.x_batch_dict['train'].to(self.device), batch_archive.y_batch_dict['train'].to(self.device))):

        # zero parameter gradients
        self.optimizer.zero_grad()

        # forward pass o:[b x c]
        o = self.models['cnn'](x)

        # loss
        loss = self.criterion(o, y)

        # backward
        loss.backward()

        # optimizer step - update params
        self.optimizer.step()

        # update batch loss collection and do print
        train_score.update_batch_losses(epoch, mini_batch, loss.item())

      # valdiation
      eval_score = self.eval_nn('validation', batch_archive)

      # update score collector
      train_score.score_dict['val_loss'][epoch], train_score.score_dict['val_acc'][epoch] = eval_score.loss, eval_score.acc

    # finish train score for time measurement
    train_score.finish()

    return train_score


  def eval_forward(self, mini_batch, x, y, z, eval_score, verbose=False):
    """
    eval forward pass
    """

    # classify
    o = self.models['cnn'](x)

    # loss
    loss = self.criterion(o, y)

    # prediction
    _, y_hat = torch.max(o.data, 1)

    # update eval score
    eval_score.update(loss, y.cpu(), y_hat.cpu(), z, o.data)

    return eval_score


  def classify_sample(self, x):
    """
    classification of a single sample presented in dim [m x f]
    """

    # input to tensor [n, c, m, f]
    x = torch.unsqueeze(torch.from_numpy(x.astype(np.float32)), 0).to(self.device)

    # no gradients for eval
    with torch.no_grad():

      # classify
      o = self.models['cnn'](x)

      # prediction
      _, y_hat = torch.max(o.data, 1)

    # int conversion
    y_hat = int(y_hat)

    # get label
    label = list(self.class_dict.keys())[list(self.class_dict.values()).index(y_hat)]

    return y_hat, o, label



class AdversarialNetHandler(NetHandler):
  """
  Adversarial Neural Network Handler
  adapted form: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
  """

  def __init__(self, nn_arch, class_dict, data_size, feature_params, encoder_model=None, decoder_model=None, use_cpu=False):

    # parent class init
    super().__init__(nn_arch, class_dict, data_size, feature_params, encoder_model=encoder_model, decoder_model=decoder_model, use_cpu=use_cpu)

    # loss criterion
    self.criterion = torch.nn.BCELoss()

    # neural network models, G-Generator, D-Discriminator
    self.init_models()

    # labels
    self.real_label = 1.
    self.fake_label = 0.

    # transfer conv weights from encoder models
    if self.encoder_model is not None: self.models['d'].transfer_params(encoder_model)
    if self.decoder_model is not None: self.models['g'].transfer_params(decoder_model)

    # update rules
    self.d_update_epochs = 1
    self.g_update_epochs = 5

    # actual counter
    self.actual_d_epoch = self.d_update_epochs
    self.actual_g_epoch = 0
    self.actual_update_epoch = 0


  def set_up_training(self, train_params):
    """
    set optimizer in training
    """

    # Setup Adam optimizers for both G and D
    self.optimizer_d = torch.optim.Adam(self.models['d'].parameters(), lr=train_params['lr_d'], betas=(train_params['beta_d'], 0.999))
    self.optimizer_g = torch.optim.Adam(self.models['g'].parameters(), lr=train_params['lr_g'], betas=(train_params['beta_g'], 0.999))


  def train_nn(self, train_params, batch_archive, callback_f=None, callback_act_epochs=10):
    """
    train adversarial nets
    """

    # setup training
    self.set_up_training(train_params)

    # Create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(32, self.models['g'].n_latent, device=self.device)

    # score collector
    train_score = AdversarialTrainScore(train_params['num_epochs'], invoker_class_name=self.__class__.__name__, k_print=batch_archive.y_batch_dict['train'].shape[0] // self.num_print_per_epoch)

    # epochs
    for epoch in range(train_params['num_epochs']):

      # fetch data samples
      for mini_batch, (x, y) in enumerate(zip(batch_archive.x_batch_dict['train'].to(self.device), batch_archive.y_batch_dict['train'].to(self.device))):

        # update models
        train_score = self.update_models(x, y, epoch, mini_batch, train_score)

      # check progess after epoch with callback function
      if callback_f is not None and not epoch % callback_act_epochs: callback_f(self.generate_samples(noise=fixed_noise, to_np=True), epoch)

    # finish train score
    train_score.finish()

    return train_score


  def update_models(self, x, y, epoch, mini_batch, train_score):
    """
    model updates
    """

    # create fakes through Generator with noise as input
    fakes = self.models['g'](torch.randn(x.shape[0], self.models['g'].n_latent, device=self.device))


    # update for each epoch
    if self.actual_update_epoch != epoch:

      # update actuals
      self.actual_update_epoch = epoch

      # discriminator update rule
      if self.actual_d_epoch: 

        self.actual_d_epoch -= 1
        if not self.actual_d_epoch: self.actual_g_epoch = self.g_update_epochs

      # generator update rule
      elif self.actual_g_epoch: 

        self.actual_g_epoch -= 1
        if not self.actual_g_epoch: self.actual_d_epoch = self.d_update_epochs


    # update discriminator
    d_loss_real, d_loss_fake = self.update_d(x, y, fakes, backward=self.actual_d_epoch > 0)

    # update generator
    g_loss_fake, g_loss_sim = self.update_g(x, fakes, backward=self.actual_g_epoch > 0)

    # update batch loss collection
    train_score.update_batch_losses(epoch, mini_batch, g_loss_fake=g_loss_fake, g_loss_sim=g_loss_sim, d_loss_real=d_loss_real, d_loss_fake=d_loss_fake)

    return train_score


  def update_d(self, x, y, fakes, backward=True):
    """
    update discriminator D with real and fake data
    """

    # zero gradients
    self.models['d'].zero_grad()
    self.optimizer_d.zero_grad()

    # create real labels
    y_real = torch.full((x.shape[0],), self.real_label, dtype=torch.float, device=self.device)

    # forward pass o:[b x c]
    o_real = self.models['d'](x).view(-1)

    # loss of D with reals
    d_loss_real = self.criterion(o_real, y_real)


    # create fake labels
    y_fake = torch.full((x.shape[0],), self.fake_label, dtype=torch.float, device=self.device)

    # fakes to D (without gradient backprop)
    o_fake = self.models['d'](fakes.detach()).view(-1)

    # loss of D with fakes
    d_loss_fake = self.criterion(o_fake, y_fake)


    # loss
    d_loss = d_loss_real + d_loss_fake

    # backward and optimization
    if backward:

      # backward
      d_loss.backward()

      # optimizer step
      self.optimizer_d.step()

    return d_loss_real.item(), d_loss_fake.item()


  def update_g(self, x, fakes, backward=True):
    """
    update generator G
    """

    # zero gradients
    self.models['g'].zero_grad()
    self.optimizer_g.zero_grad()

    # fakes should be real labels for G
    y = torch.full((fakes.shape[0],), self.real_label, dtype=torch.float, device=self.device)

    # fakes to D
    o = self.models['d'](fakes).view(-1)

    # loss of G of D with fakes
    g_loss_fake = self.criterion(o, y)

    # no similarity loss
    g_loss_sim = 0.0

    # loss
    g_loss = g_loss_fake


    # backward and optimization
    if backward:

      # backward
      g_loss.backward()

      # optimizer step
      self.optimizer_g.step()

    return g_loss_fake.item(), g_loss_sim


  def eval_forward(self, mini_batch, x, y, z, eval_score, verbose=False):
    """
    eval forward pass
    """
    
    # classify
    o = self.models['d'](x)

    # some prints
    if verbose:
      if z is not None: print("\nlabels: {}".format(z))
      print("output: {} \nactu: {} ".format(o.data, y))

    return eval_score


  def generate_samples(self, noise=None, num_samples=10, to_np=False):
    """
    generator samples from G
    """

    # generate noise if not given
    if noise is None: noise = torch.randn(num_samples, self.models['g'].n_latent, device=self.device)

    # create fakes through Generator
    with torch.no_grad():
      fakes = self.models['g'](noise).detach().cpu()

    # to numpy if necessary
    if to_np: fakes = fakes.numpy()

    return fakes



class AdversarialSimNetHandler(AdversarialNetHandler):
  """
  Adversarial Net Handler with similarity measure
  """

  def __init__(self, nn_arch, class_dict, data_size, feature_params, encoder_model=None, decoder_model=None, use_cpu=False):

    # parent class init
    super().__init__(nn_arch, class_dict, data_size, feature_params, encoder_model=encoder_model, decoder_model=decoder_model, use_cpu=use_cpu)

    # cosine similarity
    self.cos_sim = torch.nn.CosineSimilarity(dim=2, eps=1e-08)


  def update_g(self, reals, fakes, lam=5, backward=True):
    """
    update generator G
    """

    # zero gradients
    self.models['g'].zero_grad()
    self.optimizer_g.zero_grad()

    # fakes should be real labels for G
    y = torch.full((fakes.shape[0],), self.real_label, dtype=torch.float, device=self.device)

    # fakes to D
    o = self.models['d'](fakes).view(-1)

    # loss of G of D with fakes
    g_loss_fake = self.criterion(o, y)

    # similarity measure
    g_loss_sim = (1 - torch.mean(self.cos_sim(reals, fakes)))

    # loss
    g_loss = g_loss_fake + g_loss_sim * lam

    # backward and optimization
    if backward:

      # backward
      g_loss.backward()

      # optimizer step
      self.optimizer_g.step()

    return g_loss_fake.item(), g_loss_sim.item() * lam



class HybridNetHandler(NetHandler):
  """
  Hybrid Neural Network Handler
  """

  def __init__(self, nn_arch, class_dict, data_size, feature_params, encoder_model=None, decoder_model=None, use_cpu=False):

    # parent class init
    super().__init__(nn_arch, class_dict, data_size, feature_params, encoder_model=encoder_model, decoder_model=decoder_model, use_cpu=use_cpu)

    # loss criterion
    self.criterion_adv = torch.nn.BCELoss()
    self.criterion_class = torch.nn.CrossEntropyLoss()

    # neural network models init
    self.init_models()

    # labels
    self.real_label = 1.
    self.fake_label = 0.

    # transfer conv weights from encoder models
    if self.encoder_model is not None: self.models['hyb'].transfer_params(encoder_model)
    if self.decoder_model is not None: self.models['g'].transfer_params(decoder_model)

    # update rules
    self.d_update_epochs = 5
    self.g_update_epochs = 5

    # actual counter
    self.actual_d_epoch = self.d_update_epochs
    self.actual_g_epoch = self.g_update_epochs if self.g_update_epochs < 0 else 0
    self.actual_update_epoch = 0

    # cosine similarity
    self.cos_sim = torch.nn.CosineSimilarity(dim=2, eps=1e-08)


  def set_up_training(self, train_params):
    """
    set optimizer in training
    """

    # Setup Adam optimizers for both G and D
    self.optimizer_hyb = torch.optim.Adam(self.models['hyb'].parameters(), lr=train_params['lr_d'], betas=(train_params['beta_d'], 0.999))
    self.optimizer_g = torch.optim.Adam(self.models['g'].parameters(), lr=train_params['lr_g'], betas=(train_params['beta_g'], 0.999))


  def train_nn(self, train_params, batch_archive, callback_f=None, callback_act_epochs=10):
    """
    train adversarial nets
    """

    # setup training
    self.set_up_training(train_params)

    # Create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(32, self.models['g'].n_latent, device=self.device)

    # score collector
    train_score = HybridTrainScore(train_params['num_epochs'], invoker_class_name=self.__class__.__name__, k_print=batch_archive.y_batch_dict['train'].shape[0] // self.num_print_per_epoch)

    # epochs
    for epoch in range(train_params['num_epochs']):

      # fetch data samples
      for mini_batch, (x, y) in enumerate(zip(batch_archive.x_batch_dict['train'].to(self.device), batch_archive.y_batch_dict['train'].to(self.device))):

        # update models
        train_score = self.update_models(x, y, epoch, mini_batch, train_score)

      # valdiation
      eval_score = self.eval_nn('validation', batch_archive)

      # update score collector
      train_score.score_dict['val_loss'][epoch], train_score.score_dict['val_acc'][epoch] = eval_score.loss, eval_score.acc

      # check progess after epoch with callback function
      if callback_f is not None and not epoch % callback_act_epochs: callback_f(self.generate_samples(noise=fixed_noise, to_np=True), epoch)

    # finish train score
    train_score.finish()

    return train_score


  def update_models(self, x, y, epoch, mini_batch, train_score):
    """
    model updates
    """

    # create fakes through Generator with noise as input
    fakes = self.models['g'](torch.randn(x.shape[0], self.models['g'].n_latent, device=self.device))


    # update for each epoch
    if self.actual_update_epoch != epoch:

      # update actuals
      self.actual_update_epoch = epoch

      # discriminator update rule
      if self.actual_d_epoch > 0: 

        self.actual_d_epoch -= 1
        if not self.actual_d_epoch: self.actual_g_epoch = self.g_update_epochs

      # generator update rule
      elif self.actual_g_epoch > 0: 

        self.actual_g_epoch -= 1
        if not self.actual_g_epoch: self.actual_d_epoch = self.d_update_epochs


    # update discriminator
    loss_class, d_loss_real, d_loss_fake = self.update_hyb(x, y, fakes, backward=self.actual_d_epoch != 0)

    # update generator
    g_loss_fake, g_loss_sim = self.update_g(x, fakes, backward=self.actual_g_epoch != 0)

    # update batch loss collection
    train_score.update_batch_losses(epoch, mini_batch, loss_class=loss_class, d_loss_real=d_loss_real, d_loss_fake=d_loss_fake, g_loss_fake=g_loss_fake, g_loss_sim=g_loss_sim)

    return train_score


  def update_hyb(self, x, y, fakes, lam=0.5, backward=True):
    """
    update discriminator D with real and fake data
    """

    # zero gradients
    self.models['hyb'].zero_grad()
    self.optimizer_hyb.zero_grad()

    # create real labels
    y_real = torch.full((x.shape[0],), self.real_label, dtype=torch.float, device=self.device)

    # forward pass
    o_class, o_adv_real = self.models['hyb'](x)

    # loss of D with reals
    d_loss_real = self.criterion_adv(o_adv_real.view(-1), y_real)

    # loss of class label
    loss_class = self.criterion_class(o_class, y)


    # create fake labels
    y_fake = torch.full((x.shape[0],), self.fake_label, dtype=torch.float, device=self.device)

    # mixed
    y_mixed = torch.full((x.shape[0],), self.class_dict['_mixed'], dtype=torch.long, device=self.device)

    # fakes to D (without gradient backprop)
    o_mixed, o_adv_fake = self.models['hyb'](fakes.detach())

    # loss of D with fakes
    d_loss_fake = self.criterion_adv(o_adv_fake.view(-1), y_fake)

    # loss of D with fakes
    loss_mixed = self.criterion_class(o_mixed, y_mixed)

    # adv loss
    loss_adv = d_loss_real + d_loss_fake

    # calculate whole loss
    #loss = loss_class + lam * float(backward) * loss_adv
    loss = loss_class + lam * loss_mixed + lam * float(backward) * loss_adv

    # gradients for class prediction
    loss.backward()

    # optimizer step
    self.optimizer_hyb.step()

    return loss_class.item(), lam * d_loss_real.item(), lam * d_loss_fake.item()


  def update_g(self, reals, fakes, lam=4.0, backward=True):
    """
    update generator G
    """

    # zero gradients
    self.models['g'].zero_grad()
    self.optimizer_g.zero_grad()

    # fakes should be real labels for G
    y = torch.full((fakes.shape[0],), self.real_label, dtype=torch.float, device=self.device)

    # fakes to D
    o_class, o_adv_fake = self.models['hyb'](fakes)

    # loss of G of D with fakes
    g_loss_fake = self.criterion_adv(o_adv_fake.view(-1), y)

    # similarity measure
    g_loss_sim = (1 - torch.mean(self.cos_sim(reals, fakes)))

    # loss
    g_loss = g_loss_fake + g_loss_sim * lam

    # backward and optimization
    if backward:

      # backward
      g_loss.backward()

      # optimizer step
      self.optimizer_g.step()

    return g_loss_fake.item(), g_loss_sim.item() * lam


  def eval_forward(self, mini_batch, x, y, z, eval_score, verbose=False):
    """
    eval forward pass
    """

    # classify
    o_class, o_adv = self.models['hyb'](x)

    # loss
    loss = self.criterion_class(o_class, y)

    # prediction
    _, y_hat = torch.max(o_class.data, 1)

    # update eval score
    eval_score.update(loss, y.cpu(), y_hat.cpu(), z, o_class.data)

    return eval_score


  def classify_sample(self, x):
    """
    classification of a single sample with feature size dimension
    """

    # input to tensor [n, c, m, f]
    x = torch.unsqueeze(torch.from_numpy(x.astype(np.float32)), 0).to(self.device)

    # no gradients for eval
    with torch.no_grad():

      # classify
      o_class, o_adv = self.models['hyb'](x)

      # prediction
      _, y_hat = torch.max(o_class.data, 1)

    # int conversion
    y_hat = int(y_hat)

    # get label
    label = list(self.class_dict.keys())[list(self.class_dict.values()).index(y_hat)]

    return y_hat, o_class, label


  def generate_samples(self, noise=None, num_samples=10, to_np=False):
    """
    generator samples from G
    """

    # generate noise if not given
    if noise is None: noise = torch.randn(num_samples, self.models['g'].n_latent, device=self.device)

    # create fakes through Generator
    with torch.no_grad():
      fakes = self.models['g'](noise).detach().cpu()

    # to numpy if necessary
    if to_np: fakes = fakes.numpy()

    return fakes



class WavenetHandler(NetHandler):
  """
  wavenet handler
  """

  def __init__(self, nn_arch, class_dict, data_size, feature_params, encoder_model=None, decoder_model=None, use_cpu=False):

      # parent class init
      super().__init__(nn_arch, class_dict, data_size, feature_params, encoder_model=encoder_model, decoder_model=decoder_model, use_cpu=use_cpu)

      # loss criterion
      self.criterion = torch.nn.CrossEntropyLoss()

      # init models
      self.init_models()


  def set_up_training(self, train_params):
    """
    set optimizer in training
    """

    # create optimizer
    self.optimizer = torch.optim.Adam(self.models['wav'].parameters(), lr=train_params['lr'], betas=(train_params['beta'], 0.999))


  def train_nn(self, train_params, batch_archive, callback_f=None, lam=2.0):
    """
    train the neural network
    train_params: {'num_epochs': [], 'lr': [], 'momentum': []}
    """

    # setup training
    self.set_up_training(train_params)

    # score collector
    train_score = WavenetTrainScore(train_params['num_epochs'], invoker_class_name=self.__class__.__name__, k_print=batch_archive.y_batch_dict['train'].shape[0] // self.num_print_per_epoch)

    # epochs
    for epoch in range(train_params['num_epochs']):

      # update training params if necessary
      self.update_training_params(epoch, train_params)

      # fetch data samples
      for mini_batch, (x, y, t) in enumerate(zip(batch_archive.x_batch_dict['train'].to(self.device), batch_archive.y_batch_dict['train'].to(self.device), batch_archive.t_batch_dict['train'].to(self.device))):

        # zero parameter gradients
        self.optimizer.zero_grad()

        # forward pass o:[b x c]
        o_t, o_y = self.models['wav'](x)

        # loss similarity to signal
        loss_t = self.criterion(o_t, t)

        # loss for correct prediction
        loss_y = self.criterion(o_y, y)

        # loss
        loss = loss_t + loss_y * lam

        # backward
        loss.backward()

        # optimizer step - update params
        self.optimizer.step()

        # update batch loss collection
        train_score.update_batch_losses(epoch, mini_batch, loss_t=loss_t.item(), loss_y=loss_y.item() * lam)

      # valdiation
      eval_score = self.eval_nn('validation', batch_archive)

      # update score collector
      train_score.score_dict['val_loss'][epoch], train_score.score_dict['val_acc'][epoch] = eval_score.loss, eval_score.acc

    # finish train score
    train_score.finish()

    return train_score


  def eval_forward(self, mini_batch, x, y, z, eval_score, verbose=False):
    """
    forward pass
    """

    # classify
    _, o_y = self.models['wav'](x)

    # loss
    loss = self.criterion(o_y, y)

    # prediction
    _, y_hat = torch.max(o_y.data, 1)

    # update eval score
    eval_score.update(loss, y.cpu(), y_hat.cpu(), z, o_y.data)

    return eval_score


  def classify_sample(self, x):
    """
    classification of a single sample presented in dim [m x f]
    """

    # input to tensor [n, c, m, f]
    x = torch.unsqueeze(torch.from_numpy(x.astype(np.float32)), 0).to(self.device)

    # no gradients for eval
    with torch.no_grad():

      # classify
      _, o_y = self.models['wav'](x)

      # prediction
      _, y_hat = torch.max(o_y.data, 1)

    # int conversion
    y_hat = int(y_hat)

    # get label
    label = list(self.class_dict.keys())[list(self.class_dict.values()).index(y_hat)]

    return y_hat, o_y, label


if __name__ == '__main__':
  """
  handles all neural networks with training, evaluation and classifying samples 
  """

  import yaml

  from batch_archive import SpeechCommandsBatchArchive
  from audio_dataset import AudioDataset
  from plots import plot_grid_images, plot_other_grid

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # change config
  cfg['feature_params']['use_mfcc_features'] = False if cfg['ml']['nn_arch'] == 'wavenet' else True

  # set own train params
  train_params = {'batch_size': 32, 'num_epochs': 10, 'lr': 0.0001, 'lr_d': 0.0001, 'lr_g': 0.0001, 'beta': 0.9, 'beta_d': 0.9, 'beta_g': 0.9}

  # audio sets
  audio_set1 = AudioDataset(cfg['datasets']['speech_commands'], cfg['feature_params'])
  audio_set2 = AudioDataset(cfg['datasets']['my_recordings'], cfg['feature_params'])

  # create batches
  batch_archive = SpeechCommandsBatchArchive(feature_file_dict={**audio_set1.feature_file_dict, **audio_set2.feature_file_dict}, batch_size_dict={'train': train_params['batch_size'], 'test': 5, 'validation': 5, 'my': 1}, shuffle=True)
  batch_archive.create_batches() if not cfg['ml']['nn_arch'] in ['adv-experimental', 'adv-jim', 'adv-jim-label'] else batch_archive.create_batches(selected_labels=['left'])
  batch_archive.print_batch_infos()

  # create net handler
  net_handler = NetHandler(nn_arch=cfg['ml']['nn_arch'], class_dict=batch_archive.class_dict, data_size=batch_archive.data_size, feature_params=audio_set1.feature_params, use_cpu=cfg['ml']['use_cpu'])
  print(net_handler.models)

  # training
  net_handler.train_nn(train_params, batch_archive=batch_archive)

  # test
  net_handler.eval_nn('test', batch_archive=batch_archive, collect_things=False, verbose=False)

  # classify sample
  y_hat, o, label = net_handler.classify_sample(torch.randn(net_handler.data_size).numpy())

  # print classify result
  print("classify: [{}]\noutput: [{}]\nlabel: [{}]".format(y_hat, o, label))

  # count parameters
  count_dict = net_handler.count_params_and_mults()

  print("count dict: ", count_dict)
  for k, l in count_dict.items(): print("{} sum: {}".format(k, np.sum(l)))