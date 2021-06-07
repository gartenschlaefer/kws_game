"""
Score classes for training and evaluation
"""

import numpy as np
import time


class TrainScore():
  """
  collection of scores
  """

  def __init__(self, num_epochs, invoker_class_name='', k_print=10):

    # arguments
    self.num_epochs = num_epochs
    self.k_print= k_print

    # print safety
    self.do_print_anyway, self.k_print = (True, 1) if not self.k_print else (False, self.k_print)

    # dictionaries
    self.score_dict = {'score_class': self.__class__.__name__, 'val_loss': np.zeros(self.num_epochs), 'val_acc': np.zeros(self.num_epochs), 'time_usage': None}
    self.batch_dict = {}

    # specific dictionary update
    self.init_dictionaries()

    # print message
    print("\n--Training starts {}".format(invoker_class_name))

    # start time init
    self.start_time = time.time()


  def define_loss_types(self):
    """
    define loss types
    """
    return ['train_loss']


  def init_dictionaries(self):
    """
    init dictionaries like score dict, etc.
    """

    # loss types
    loss_types = self.define_loss_types()

    # score dictionary
    [self.score_dict.update({k: np.zeros(self.num_epochs)}) for k in loss_types]

    # batch losses
    [self.batch_dict.update({k + '_batch': 0.0}) for k in loss_types]


  def reset_batch_losses(self):
    """
    reset batch losses to zero
    """
    for k in self.batch_dict.keys(): self.batch_dict[k] = 0.0


  def update_batch_losses(self, epoch, mini_batch, loss):
    """
    update losses
    """

    # update losses
    self.score_dict['train_loss'][epoch] += loss
    self.batch_dict['train_loss_batch'] += loss

    # print loss
    if mini_batch % self.k_print == self.k_print-1 or self.do_print_anyway:

      # do print
      self.print_train_info(epoch, mini_batch)

      # reset batch losses
      self.reset_batch_losses()


  def print_train_info(self, epoch, mini_batch):
    """
    print some training info
    """
    print('epoch: {}, mini-batch: {}, loss: [{:.5f}]'.format(epoch + 1, mini_batch + 1, self.batch_dict['train_loss_batch']))


  def finish(self):
    """
    finish training
    """

    # log time
    self.score_dict['time_usage'] = time.time() - self.start_time 
    print('--Training finished')



class AdversarialTrainScore(TrainScore):
  """
  collection of scores
  """

  def define_loss_types(self):
    """
    define loss types
    """
    return ['g_loss_fake', 'g_loss_sim', 'd_loss_real', 'd_loss_fake']


  def update_batch_losses(self, epoch, mini_batch, g_loss_fake, g_loss_sim, d_loss_real, d_loss_fake):
    """
    update losses
    """

    # update losses
    self.score_dict['g_loss_fake'][epoch] += g_loss_fake
    self.batch_dict['g_loss_fake_batch'] += g_loss_fake

    self.score_dict['g_loss_sim'][epoch] += g_loss_sim
    self.batch_dict['g_loss_sim_batch'] += g_loss_sim

    self.score_dict['d_loss_real'][epoch] += d_loss_real
    self.batch_dict['d_loss_real_batch'] += d_loss_real

    self.score_dict['d_loss_fake'][epoch] += d_loss_fake
    self.batch_dict['d_loss_fake_batch'] += d_loss_fake

    # print loss
    if mini_batch % self.k_print == self.k_print-1 or self.do_print_anyway:

      # do print
      self.print_train_info(epoch, mini_batch)

      # reset batch losses
      self.reset_batch_losses()


  def print_train_info(self, epoch, mini_batch):
    """
    print some training info
    """
    print('epoch: {}, mini-batch: {}, G loss fake: [{:.5f}], G loss sim: [{:.5f}], D loss real: [{:.5f}], D loss fake: [{:.5f}]'.format(epoch + 1, mini_batch + 1, self.batch_dict['g_loss_fake_batch'], self.batch_dict['g_loss_sim_batch'], self.batch_dict['d_loss_real_batch'], self.batch_dict['d_loss_fake_batch']))



class WavenetTrainScore(TrainScore):
  """
  collection of scores for wavenets
  """

  def define_loss_types(self):
    """
    define loss types
    """
    return ['loss_t', 'loss_y']


  def update_batch_losses(self, epoch, mini_batch, loss_t, loss_y):
    """
    update losses
    """

    # update losses
    self.score_dict['loss_t'][epoch] += loss_t
    self.batch_dict['loss_t_batch'] += loss_t

    self.score_dict['loss_y'][epoch] += loss_y
    self.batch_dict['loss_y_batch'] += loss_y

    # print loss
    if mini_batch % self.k_print == self.k_print-1 or self.do_print_anyway:

      # do print
      self.print_train_info(epoch, mini_batch)

      # reset batch losses
      self.reset_batch_losses()


  def print_train_info(self, epoch, mini_batch):
    """
    print some training info
    """
    print('epoch: {}, mini-batch: {}, loss_t: [{:.5f}], loss_y: [{:.5f}]'.format(epoch + 1, mini_batch + 1, self.batch_dict['loss_t_batch'], self.batch_dict['loss_y_batch']))



class EvalScore():
  """
  typical scores for evaluation
  """

  def __init__(self, eval_set_name='', collect_things=False):

    # arguments
    self.eval_set_name = eval_set_name
    self.collect_things = collect_things

    # init vars
    self.loss, self.correct, self.total, self.acc, self.cm = 0.0, 0, 0, 0.0, None

    # all actual and predicted labels from batches
    self.y_all, self.y_hat_all = [], []
    
    # all outputs and names
    self.o_all, self.z_all = [], []

    # wrongly classified sample list
    self.wrong_list = []


  def update(self, loss, y, y_hat, z, o):
    """
    update score (batch update)
    """

    # loss update
    self.loss += loss

    # add total amount of prediction
    self.total += y.size(0)

    # check if correctly predicted
    self.correct += (y_hat == y).sum().item()

    # collect everything (not efficient so keep care of collect flag)
    if self.collect_things:
      wrong_vector = (y_hat != y).numpy()
      if wrong_vector.any(): self.wrong_list.append(z[wrong_vector].tolist())
      self.y_all.append(y.tolist())
      self.y_hat_all.append(y_hat.tolist())
      self.o_all.append(o.tolist())
      self.z_all.append(z.tolist())


  def finish(self):
    """
    finishing procedure
    """

    from sklearn.metrics import confusion_matrix

    # safety for devision
    if self.total == 0: self.total = 1

    # accuracy
    self.acc = 100 * self.correct / self.total

    # print eval results
    self.info_log(do_print=True)

    # collecting check
    if not self.collect_things or not len(self.y_all): return

    # confusion matrix
    self.cm = confusion_matrix(np.concatenate(self.y_all), np.concatenate(self.y_hat_all))


  def info_log(self, do_print=True):
    """
    print evaluation outcome
    """

    # message
    eval_log = "Eval{}: correct: [{} / {}] acc: [{:.4f}] with loss: [{:.4f}]\n".format(' ' + self.eval_set_name, self.correct, self.total, self.acc, self.loss)

    # print
    if do_print: print(eval_log)

    return eval_log


  def info_detail_log(self, arch, param_string, train_params):
    """
    detail log
    """
    return "Eval{} on arch: [{}], audio set param string: [{}], train_params: {}, correct: [{} / {}] acc: [{:.4f}] with loss: [{:.4f}]".format(' ' + self.eval_set_name, arch, param_string, train_params, self.correct, self.total, self.acc, self.loss)


  def info_collected(self, arch, param_string, train_params, info_file=None, do_print=False):
    """
    show infos
    """

    # collecting check
    if not self.collect_things:
      print("collection flag was not set")
      return

    # formating of outputs
    self.o_all = [[['{:.4e}'.format(iii) for iii in ii] for ii in i] for i in self.o_all]

    # file print
    if info_file is not None:
      with open(info_file, 'a') as f:
        print("\n#--\n{}\nacc: [{:.4f}]".format(self.info_detail_log(arch, param_string, train_params), self.acc), file=f)
        print("\nwrongs: ", np.concatenate(self.wrong_list) if len(self.wrong_list) else 'none', file=f)
        for y, y_hat, o, z in zip(self.y_all, self.y_hat_all, self.o_all, self.z_all): print("\nz: {}\noutput: {}\npred: {}, actu: {}, \t corr: {} ".format(z, o, y_hat, y, (np.array(y_hat) == np.array(y)).astype('int')), file=f)

    # command line print
    if do_print:
      print("\n#--\n{}\nacc: [{:.4f}]".format(self.info_detail_log(arch, param_string, train_params), self.acc))
      print("\nwrongs: ", np.concatenate(self.wrong_list) if len(self.wrong_list) else 'none')
      for y, y_hat, o, z in zip(self.y_all, self.y_hat_all, self.o_all, self.z_all): print("\nz: {}\noutput: {}\npred: {}, actu: {}, \t corr: {} ".format(z, o, y_hat, y, (np.array(y_hat) == np.array(y)).astype('int')))


if __name__ == '__main__':
  """
  main
  """

  #train_score = TrainScore(5)
  #train_score = AdversarialTrainScore(5)
  train_score = WavenetTrainScore(5)

  print("score dict: ", train_score.score_dict)

  train_score.print_train_info(1, 9)