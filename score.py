"""
Score classes for training and evaluation
"""

import numpy as np


class TrainScore():
  """
  collection of scores
  """

  def __init__(self, num_epochs, is_adv=False):

    # arguments
    self.num_epochs = num_epochs
    self.is_adv = is_adv

    # collect losses over epochs
    self.batch_loss = 0.0
    self.train_loss = np.zeros(self.num_epochs)
    self.val_loss = np.zeros(self.num_epochs)
    self.val_acc = np.zeros(self.num_epochs)

    # adversarial losses
    self.g_batch_loss_fake = None
    self.g_batch_loss_sim = None
    self.d_batch_loss_real = None
    self.d_batch_loss_fake = None
    self.g_train_loss = None
    self.d_train_loss_real = None
    self.d_train_loss_fake = None

    # init if activated
    if self.is_adv:
      self.g_loss_fake = np.zeros(self.num_epochs)
      self.g_loss_sim = np.zeros(self.num_epochs)
      self.d_loss_real = np.zeros(self.num_epochs)
      self.d_loss_fake = np.zeros(self.num_epochs)

    # reset batch losses
    self.reset_batch_losses()

    # time score
    self.time_usage = None


  def reset_batch_losses(self):
    """
    reset batch losses to zero
    """

    # reset batch loss
    self.batch_loss = 0.0

    # for adversarial networks
    if self.is_adv:
      self.g_batch_loss_fake = 0.0
      self.g_batch_loss_lim = 0.0
      self.d_batch_loss_real = 0.0
      self.d_batch_loss_fake = 0.0


  def update_batch_losses(self, epoch, loss, g_loss_fake=0.0, g_loss_sim=0.0, d_loss_real=0.0, d_loss_fake=0.0):
    """
    update losses
    """

    # update losses
    self.batch_loss = loss
    self.train_loss[epoch] += loss 

    # adversarial net
    if self.is_adv:

      # batch loss for score
      self.g_batch_loss_fake = g_loss_fake
      self.g_batch_loss_fake = g_loss_sim
      self.d_batch_loss_real = d_loss_real
      self.d_batch_loss_fake = d_loss_fake

      self.g_loss_fake[epoch] += g_loss_fake
      self.g_loss_sim[epoch] += g_loss_sim
      self.d_loss_real[epoch] += d_loss_real
      self.d_loss_fake[epoch] += d_loss_fake



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
    if not self.collect_things: return

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


  def info_collected(self, info_file=None, do_print=False):
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
        print("\n--eval {}".format(self.eval_set_name), file=f)
        print("wrongs: ", np.concatenate(self.wrong_list), file=f)
        for y, y_hat, o, z in zip(self.y_all, self.y_hat_all, self.o_all, self.z_all): print("\nz: {}\noutput: {}\npred: {}, actu: {}, \t corr: {} ".format(z, o, y_hat, y, (np.array(y_hat) == np.array(y)).astype('int')), file=f)

    # command line print
    if do_print:
      print("\n--eval {}".format(self.eval_set_name))
      print("wrongs: ", np.concatenate(self.wrong_list))
      for y, y_hat, o, z in zip(self.y_all, self.y_hat_all, self.o_all, self.z_all): print("\nz: {}\noutput: {}\npred: {}, actu: {}, \t corr: {} ".format(z, o, y_hat, y, (np.array(y_hat) == np.array(y)).astype('int')))

