"""
Handling Neural Networks
"""

import numpy as np
import torch
import time

from conv_nets import ConvNetTrad, ConvNetFstride4, ConvNetExperimental, ConvEncoderClassifierNet, ConvStackedEncodersNet, ConvLatentClassifier
from adversarial_nets import G_experimental, D_experimental
from score import TrainScore, EvalScore


class NetHandler():
  """
  Neural Network Handler with general functionalities and interfaces
  """

  def __new__(cls, nn_arch, class_dict, data_size, encoder_model=None, decoder_model=None, use_cpu=False):

    # adversarial handler
    if nn_arch in ['adv-experimental']:
      for child_cls in cls.__subclasses__():
        if child_cls.__name__ == 'AdversarialNetHandler':
          return super().__new__(AdversarialNetHandler)

    # experimental2
    if nn_arch in ['adv-experimental3', 'adv-collected-encoder', 'adv-lim-encoder', 'adv-lim-encoder-6']:
      for child_cls in cls.__subclasses__():
        for child_child_cls in child_cls.__subclasses__():
          if child_child_cls.__name__ == 'AdversarialNetHandlerExperimental':
            return super().__new__(AdversarialNetHandlerExperimental)

    # cnn handler
    elif nn_arch in ['conv-trad', 'conv-fstride', 'conv-experimental']:
      for child_cls in cls.__subclasses__():
        if child_cls.__name__ == 'CnnHandler':
          return super().__new__(CnnHandler)

    # conv encoder handler
    elif nn_arch in ['conv-encoder', 'conv-encoder-fc1', 'conv-encoder-fc3', 'conv-lim-encoder', 'conv-latent']:
      for child_cls in cls.__subclasses__():
        if child_cls.__name__ == 'ConvEncoderNetHandler':
          return super().__new__(ConvEncoderNetHandler)

        for child_child_cls in child_cls.__subclasses__():
          if child_child_cls.__name__ == 'ConvEncoderNetHandler':
            return super().__new__(ConvEncoderNetHandler)

    # handler specific
    return super().__new__(cls)


  def __init__(self, nn_arch, class_dict, data_size, encoder_model=None, decoder_model=None, use_cpu=False):

    # arguments
    self.nn_arch = nn_arch
    self.class_dict = class_dict
    self.data_size = data_size
    self.encoder_model = encoder_model
    self.decoder_model = decoder_model
    self.use_cpu = use_cpu

    # vars
    self.n_classes = len(self.class_dict)
    self.num_print_per_epoch = 2

    # set device
    self.device = torch.device("cuda:0" if (torch.cuda.is_available() and not self.use_cpu) else "cpu")

    # print msg
    #print("device: ", self.device)
    #if torch.cuda.is_available() and not self.use_cpu: print("use gpu: ", torch.cuda.get_device_name(self.device))

    # models dictionary key: name, value: model
    self.models = {}


  def init_models(self):
    """
    instantiate the requested models and sent them to the device
    """

    # traditionals
    if self.nn_arch == 'conv-trad': self.models = {'cnn':ConvNetTrad(self.n_classes, self.data_size)}
    elif self.nn_arch == 'conv-fstride': self.models = {'cnn':ConvNetFstride4(self.n_classes, self.data_size)}
    elif self.nn_arch == 'conv-experimental': self.models = {'cnn':ConvNetExperimental(self.n_classes, self.data_size)}
    
    # adversarials
    elif self.nn_arch == 'adv-experimental': self.models = {'g':G_experimental(self.n_classes, self.data_size), 'd':D_experimental(self.n_classes, self.data_size)}
    elif self.nn_arch == 'adv-experimental3': self.models = {'g':G_experimental(self.n_classes, self.data_size), 'd':D_experimental(self.n_classes, self.data_size, out_dim=1)}
    elif self.nn_arch == 'adv-collected-encoder': self.models = {'g':G_experimental(self.n_classes, self.data_size, net_class='label-collect-encoder'), 'd':D_experimental(self.n_classes, self.data_size, net_class='label-collect-encoder')}
    
    # cnns
    elif self.nn_arch == 'conv-encoder': self.models = {'cnn':ConvEncoderClassifierNet(self.n_classes, self.data_size, net_class='label-collect-encoder')}
    elif self.nn_arch == 'conv-encoder-stacked': self.models = {'cnn':ConvStackedEncodersNet(self.n_classes, self.data_size, self.encoder_model)}
    elif self.nn_arch == 'conv-latent': self.models = {'cnn':ConvLatentClassifier(self.n_classes, self.data_size)}

    # selected fc
    elif self.nn_arch == 'conv-encoder-fc1': self.models = {'cnn':ConvEncoderClassifierNet(self.n_classes, self.data_size, net_class='lim-encoder-6', fc_layer_type='fc1')}
    elif self.nn_arch == 'conv-encoder-fc3': self.models = {'cnn':ConvEncoderClassifierNet(self.n_classes, self.data_size, net_class='lim-encoder-6', fc_layer_type='fc3')}
    
    # limited encoders adv
    elif self.nn_arch == 'adv-lim-encoder': self.models = {'g':G_experimental(self.n_classes, self.data_size, net_class='lim-encoder'), 'd':D_experimental(self.n_classes, self.data_size, net_class='lim-encoder')}
    elif self.nn_arch == 'adv-lim-encoder-6': self.models = {'g':G_experimental(self.n_classes, self.data_size, net_class='lim-encoder-6'), 'd':D_experimental(self.n_classes, self.data_size, net_class='lim-encoder-6')}
    
    # limited encoder conv
    elif self.nn_arch == 'conv-lim-encoder': self.models = {'cnn':ConvEncoderClassifierNet(self.n_classes, self.data_size, net_class='lim-encoder')}
    
    # not found
    else: print("***Network Architecture not found!")

    # send models to device
    self.models = dict((k, model.to(self.device)) for k, model in self.models.items())


  def get_nn_arch_has_conv_coder(self):
    """
    conv coder flags
    """
    enc = self.nn_arch in  ['conv-encoder', 'conv-encoder-fc1', 'conv-encoder-fc3', 'conv-lim-encoder', 'conv-latent', 'adv-experimental', 'adv-experimental3', 'adv-collected-encoder', 'adv-lim-encoder', 'adv-lim-encoder-6']
    dec = self.nn_arch in  ['adv-experimental', 'adv-experimental3', 'adv-collected-encoder', 'adv-lim-encoder', 'adv-lim-encoder-6']
    return enc, dec
    

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
      if train_score.is_adv:
        print('epoch: {}, mini-batch: {}, G loss fake: [{:.5f}], D loss real: [{:.5f}], D loss fake: [{:.5f}]'.format(epoch + 1, mini_batch + 1, train_score.g_batch_loss_fake / k_print, train_score.d_batch_loss_real / k_print, train_score.d_batch_loss_fake / k_print))

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
      return False

    # load models
    for model_file, (k, model) in zip(model_files, self.models.items()):
      print("load model: {}, net handler model: {}".format(model_file, k))
      model.load_state_dict(torch.load(model_file))

    return True


  def save_models(self, model_files, encoder_model_file=None, decoder_model_file=None, encoder_class_name='ConvEncoder', decoder_class_name='ConvDecoder'):
    """
    saves model
    """

    # safety check
    if len(model_files) != len(self.models):
      print("***len of model file names is not equal length of models")
      return

    # load models
    for model_file, (k, model) in zip(model_files, self.models.items()):

      # save model
      torch.save(model.state_dict(), model_file)
      print("save model: {}, net handler model: {}".format(model_file, k))

      # try:
      #   print("save model: {}, net handler model: {}".format(model_file, k))
      #   torch.save(model.state_dict(), model_file)

      # except:
      #   print("\n***could not save model!!!\n")

      # skip if encoder model file is None
      if encoder_model_file is None and decoder_model_file is None: continue

      # go through all modules
      for module in model.children():

        # if module is encoder class
        if module.__class__.__name__ == encoder_class_name:

          # save and print info
          torch.save(module.state_dict(), encoder_model_file)
          print("save {} to file: {}".format(encoder_class_name, encoder_model_file))

        # if module is encoder class
        elif module.__class__.__name__ == decoder_class_name:

          # save and print info
          torch.save(module.state_dict(), decoder_model_file)
          print("save {} to file: {}".format(decoder_class_name, decoder_model_file))   


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


  def eval_nn(self, eval_set, batch_archive, collect_things=False, verbose=False):
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

    # eval set selection
    if eval_set == 'val': x_eval, y_eval, z_eval = batch_archive.x_val, batch_archive.y_val, batch_archive.z_val
    elif eval_set == 'test': x_eval, y_eval, z_eval = batch_archive.x_test, batch_archive.y_test, batch_archive.z_test
    elif eval_set == 'my': x_eval, y_eval, z_eval = batch_archive.x_my, batch_archive.y_my, batch_archive.z_my
    else: print("wrong usage of eval nn, select eval_set one out of ['val', 'test', 'my']")

    return x_eval, y_eval, z_eval


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

  def __init__(self, nn_arch, class_dict, data_size, encoder_model=None, decoder_model=None, use_cpu=False):

    # parent class init
    super().__init__(nn_arch, class_dict, data_size, encoder_model=encoder_model, decoder_model=decoder_model, use_cpu=use_cpu)

    # loss criterion
    self.criterion = torch.nn.CrossEntropyLoss()

    # init models
    self.init_models()

    # load encoder decoder models
    self.load_coder_models()


  def load_coder_models(self):

    # using encoder or decoder model, one should be not None
    if bool(self.encoder_model is None) ^ bool(self.decoder_model is None):

      # encoder model
      if self.encoder_model is not None: self.models['cnn'].conv_encoder.load_state_dict(self.encoder_model.conv_encoder.state_dict())

      # transfer conv weights from encoder models
      elif self.decoder_model is not None: 
        #print("decoder: ", decoder_model.state_dict().keys())
        self.models['cnn'].conv_encoder.transfer_decoder_weights(self.decoder_model.conv_decoder)

        # for latent space (might be removed)
        with torch.no_grad():
          if 'fc_latent.weight' in self.models['cnn'].state_dict().keys() and 'fc1.weight' in self.decoder_model.state_dict().keys(): 
            self.models['cnn'].state_dict()['fc_latent.weight'][:] = decoder_model.state_dict()[param_tensor].T
            print("latent yeah")


  def set_up_training(self, train_params):
    """
    set optimizer in training
    """

    # create optimizer
    #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=train_params['lr'], momentum=train_params['momentum'])
    self.optimizer = torch.optim.Adam(self.models['cnn'].parameters(), lr=train_params['lr'])


  def train_nn(self, train_params, batch_archive, callback_f=None):
    """
    train the neural network
    train_params: {'num_epochs': [], 'lr': [], 'momentum': []}
    """

    # setup training
    self.set_up_training(train_params)

    # score collector
    train_score = TrainScore(train_params['num_epochs'])

    print("\n--Training starts:")

    # start time
    start_time = time.time()

    # epochs
    for epoch in range(train_params['num_epochs']):

      # update training params if necessary
      self.update_training_params(epoch, train_params)

      # TODO: do this with loader function from pytorch (maybe or not)
      # fetch data samples
      for i, (x, y) in enumerate(zip(batch_archive.x_train.to(self.device), batch_archive.y_train.to(self.device))):

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


  def eval_nn(self, eval_set, batch_archive, collect_things=False, verbose=False):
    """
    evaluation of nn
    use eval_set out of ['val', 'test', 'my']
    """

    # eval mode
    self.set_eval_mode()

    # select the evaluation set
    x_eval, y_eval, z_eval = self.eval_select_set(eval_set, batch_archive)

    # if set does not exist
    if x_eval is None or y_eval is None:
      print("no eval set found")
      return EvalScore(eval_set_name=eval_set, collect_things=collect_things)

    # init score
    eval_score = EvalScore(eval_set_name=eval_set, collect_things=collect_things)

    # no gradients for eval
    with torch.no_grad():

      # load data
      for i, (x, y, z) in enumerate(zip(x_eval.to(self.device), y_eval.to(self.device), z_eval)):

        # classify
        o = self.models['cnn'](x)

        # loss
        loss = self.criterion(o, y)

        # prediction
        _, y_hat = torch.max(o.data, 1)

        # update eval score
        eval_score.update(loss, y.cpu(), y_hat.cpu(), z, o.data)

    # finish up scores
    eval_score.finish()

    # train mode
    self.set_train_mode()

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

  def __init__(self, nn_arch, class_dict, data_size, encoder_model=None, decoder_model=None, use_cpu=False):

    # parent class init
    super().__init__(nn_arch, class_dict, data_size, encoder_model=encoder_model, decoder_model=decoder_model, use_cpu=use_cpu)

    # loss criterion
    self.criterion = torch.nn.BCELoss()

    # neural network models, G-Generator, D-Discriminator
    self.init_models()

    # labels
    self.real_label = 1.
    self.fake_label = 0.

    # transfer conv weights from encoder models
    if self.encoder_model is not None: self.models['d'].conv_encoder.load_state_dict(encoder_model.conv_encoder.state_dict())
    if self.decoder_model is not None: self.models['g'].conv_decoder.load_state_dict(decoder_model.conv_decoder.state_dict())


  def set_up_training(self, train_params):
    """
    set optimizer in training
    """

    # Setup Adam optimizers for both G and D
    self.optimizer_d = torch.optim.Adam(self.models['d'].parameters(), lr=train_params['lr'], betas=(train_params['beta'], 0.999))
    self.optimizer_g = torch.optim.Adam(self.models['g'].parameters(), lr=train_params['lr'], betas=(train_params['beta'], 0.999))


  def train_nn(self, train_params, batch_archive, callback_f=None):
    """
    train adversarial nets
    """

    # setup training
    self.set_up_training(train_params)

    # Create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(32, self.models['g'].n_latent, device=self.device)

    # score collector
    train_score = TrainScore(train_params['num_epochs'], is_adv=True)

    print("\n--Training starts:")

    # start time
    start_time = time.time()

    # epochs
    for epoch in range(train_params['num_epochs']):

      # fetch data samples
      for batch_index, (x, y) in enumerate(zip(batch_archive.x_train.to(self.device), batch_archive.y_train.to(self.device))):

        # update models
        train_score = self.update_models(x, y, batch_archive.class_dict, epoch, train_score)

        # print some infos
        self.print_train_info(epoch, batch_index, train_score, k_print=batch_archive.y_train.shape[0] // self.num_print_per_epoch)
        train_score.reset_batch_losses()

      # check progess after epoch with callback function
      if callback_f is not None:
        callback_f(self.generate_samples(noise=fixed_noise, to_np=True))

    print('--Training finished')

    # log time
    train_score.time_usage = time.time() - start_time 

    return train_score


  def update_models(self, x, y, class_dict, epoch, train_score):
    """
    model updates
    """

    # create fakes through Generator with noise as input
    fakes = self.models['g'](torch.randn(x.shape[0], self.models['g'].n_latent, device=self.device))

    # update discriminator
    d_loss_real, d_loss_fake = self.update_d(x, y, class_dict, fakes)

    # update generator
    g_loss_fake = self.update_g(fakes)

    # update batch loss collection
    train_score.update_batch_losses(epoch, loss=0.0, g_loss_fake=g_loss_fake.item(), d_loss_real=d_loss_real.item(), d_loss_fake=d_loss_fake.item())

    return train_score


  def update_d(self, x, y, class_dict, fakes):
    """
    update discriminator D with real and fake data
    """

    # zero parameter gradients
    self.models['d'].zero_grad()
    self.optimizer_d.zero_grad()

    # create real labels
    y = torch.full((x.shape[0],), self.real_label, dtype=torch.float, device=self.device)

    # forward pass o:[b x c]
    o = self.models['d'](x).view(-1)

    # loss of D with reals
    d_loss_real = self.criterion(o, y)
    d_loss_real.backward()

    # create fake labels
    y.fill_(self.fake_label)

    # fakes to D (without gradient backprop)
    o = self.models['d'](fakes.detach()).view(-1)

    # loss of D with fakes
    d_loss_fake = self.criterion(o, y) * 1
    d_loss_fake.backward()

    # optimizer step
    self.optimizer_d.step()

    return d_loss_real, d_loss_fake


  def update_g(self, fakes):
    """
    update generator G
    """

    # zero gradients
    self.models['g'].zero_grad()

    # fakes should be real labels for G
    y = torch.full((fakes.shape[0],), self.real_label, dtype=torch.float, device=self.device)

    # fakes to D
    o = self.models['d'](fakes).view(-1)

    # loss of G of D with fakes
    g_loss_fake = self.criterion(o, y)
    g_loss_fake.backward()

    # optimizer step
    self.optimizer_g.step()

    return g_loss_fake


  def eval_nn(self, eval_set, batch_archive, collect_things=False, verbose=False):
    """
    evaluation of nn
    use eval_set out of ['val', 'test', 'my']
    """

    # eval mode
    self.set_eval_mode()

    # select the evaluation set
    x_eval, y_eval, z_eval = self.eval_select_set(eval_set, batch_archive)

    # init score
    eval_score = EvalScore(collect_things=False)

    # if set does not exist
    if x_eval is None or y_eval is None:
      print("no eval set found")
      return eval_score

    # no gradients for eval
    with torch.no_grad():

      # load data
      for i, (x, y) in enumerate(zip(x_eval.to(self.device), y_eval.to(self.device))):

        # classify
        o = self.models['d'](x)

        # some prints
        if verbose:
          if z_eval is not None:
            print("\nlabels: {}".format(z_eval[i]))
          print("output: {} \nactu: {} ".format(o.data, y))

    # finish up scores
    eval_score.finish()

    # train mode
    self.set_train_mode()

    return eval_score


  def generate_samples(self, noise=None, num_samples=10, to_np=False):
    """
    generator samples from G
    """

    # generate noise if not given
    if noise is None:
      noise = torch.randn(num_samples, self.models['g'].n_latent, device=self.device)

    # create fakes through Generator
    with torch.no_grad():
      fakes = self.models['g'](noise).detach().cpu()

    # to numpy if necessary
    if to_np:
      fakes = fakes.numpy()

    return fakes



class AdversarialNetHandlerExperimental(AdversarialNetHandler):
  """
  Experimental Adversarial Net Handler
  """

  def __init__(self, nn_arch, class_dict, data_size, encoder_model=None, decoder_model=None, use_cpu=False):

    # parent class init
    super().__init__(nn_arch, class_dict, data_size, encoder_model=encoder_model, decoder_model=decoder_model, use_cpu=use_cpu)

    # loss criterion
    self.criterion = torch.nn.BCELoss()

    # labels
    self.real_label = 1.
    self.fake_label = 0.

    # cosine similarity
    self.cos_sim = torch.nn.CosineSimilarity(dim=2, eps=1e-08)


  def update_models(self, x, y, class_dict, epoch, train_score):
    """
    model updates
    """

    # create fakes through Generator with noise as input
    fakes = self.models['g'](torch.randn(x.shape[0], self.models['g'].n_latent, device=self.device))

    # update discriminator
    d_loss_real, d_loss_fake = self.update_d(x, y, class_dict, fakes)

    # update generator
    g_loss_fake, g_loss_sim = self.update_g(x, fakes, class_dict)

    # update batch loss collection
    train_score.update_batch_losses(epoch, loss=0.0, g_loss_fake=g_loss_fake, g_loss_sim=g_loss_sim, d_loss_real=d_loss_real, d_loss_fake=d_loss_fake)

    return train_score


  def update_d(self, x, y, class_dict, fakes):
    """
    update discriminator D with real and fake data
    """
 
    # zero parameter gradients
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

    # backward
    d_loss.backward()

    # optimizer step
    self.optimizer_d.step()

    return d_loss_real.item(), d_loss_fake.item()


  def update_g(self, reals, fakes, class_dict, lam=6):
    """
    update generator G
    """

    # zero gradients
    self.models['g'].zero_grad()

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

    # backward
    g_loss.backward()

    # optimizer step
    self.optimizer_g.step()

    return g_loss_fake.item(), g_loss_sim.item() * lam



class ConvEncoderNetHandler(CnnHandler):
  """
  Neural Network Handler for Convolutional Encoder Network
  the conv encoders are pre-trained and only a consecutive classifier network is trained
  """

  def set_up_training(self, train_params):
    """
    set optimizer in training
    """

    # create optimizer
    #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=train_params['lr'], momentum=train_params['momentum'])
    self.optimizer = torch.optim.Adam(self.models['cnn'].parameters(), lr=train_params['lr'])
    #self.optimizer = torch.optim.Adam(self.models['cnn'].classifier_net.parameters(), lr=train_params['lr'])
    #self.models['cnn'].encoder_model.eval()


  def update_training_params(self, epoch, train_params):
    """
    update training parameters upon epoch
    """
    
    if epoch == 0:
      print("update training params")
      #self.optimizer = torch.optim.Adam(self.models['cnn'].encoder_model.parameters(), lr=train_params['lr'])



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

  # audio sets
  audio_set1 = AudioDataset(cfg['datasets']['speech_commands'], cfg['feature_params'])
  audio_set2 = AudioDataset(cfg['datasets']['my_recordings'], cfg['feature_params'])

  # create batches
  batch_archive = SpeechCommandsBatchArchive(audio_set1.feature_files + audio_set2.feature_files, batch_size=32, batch_size_eval=5)

  # reduce to label and add noise
  #batch_archive.reduce_to_label('up')
  #batch_archive.add_noise_data(shuffle=True)

  print("data: ", batch_archive.data_size)
  print("classes: ", batch_archive.class_dict)

  # create an cnn handler
  net_handler = NetHandler(nn_arch=cfg['ml']['nn_arch'], class_dict=batch_archive.class_dict, data_size=batch_archive.data_size, use_cpu=cfg['ml']['use_cpu'])
  print(net_handler.models)

  # training
  #net_handler.train_nn(cfg['ml']['train_params'], batch_archive=batch_archive)

  # validation
  #net_handler.eval_nn(eval_set='val', batch_archive=batch_archive, collect_things=False, verbose=False)

  # classify sample
  y_hat, o, label = net_handler.classify_sample(np.random.randn(net_handler.data_size[0], net_handler.data_size[1], net_handler.data_size[2]))

  # print classify result
  print("classify: [{}]\noutput: [{}]\nlabel: [{}]".format(y_hat, o, label))

  # count parameters
  count_dict = net_handler.count_params_and_mults()

  print("count dict: ", count_dict)
  for k, l in count_dict.items(): print("sum: ", np.sum(l))