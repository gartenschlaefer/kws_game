"""
Handling Neural Networks
"""

import numpy as np
import torch

from sklearn.metrics import confusion_matrix


class NetHandler():
  """
  Neural Network Handler
  """

  def __init__(self, nn_arch, batch_archiv, pre_trained_model_path=None):

  	# neural net architecture (see in config which are available : string)
  	self.nn_arch = nn_arch

  	# training set
  	self.batch_archiv = batch_archiv

  	# pre trained models
  	self.pre_trained_model_path = pre_trained_model_path

  	# neural network model
  	self.model = get_nn_model()


	def get_nn_model(self):
	  """
	  simply get the desired nn model
	  """

	  # select network architecture
	  if self.nn_arch == 'conv-trad':

	    # traditional conv-net
	    self.model = ConvNetTrad(batch_archiv.n_classes)

	  elif self.nn_arch == 'conv-fstride':

	    # limited multipliers conv-net
	    self.model = ConvNetFstride4(batch_archiv.n_classes)

	  # did not find architecture
	  else:

	    print("Network Architecture not found, uses: conf-trad")

	    # traditional conv-net
	    self.model = ConvNetTrad(batch_archiv.n_classes)


	def get_pretrained_model(self, pre_trained_model_path):
	  """
	  get pretrained model
	  """

	  # same model
	  if self.pre_trained_model_path is None:
	    return

	  # load model
	  try:
	    print("load model: ", pre_trained_model_path)
	    self.model.load_state_dict(torch.load(pre_trained_model_path))

	  except:
	    print("could not load pre-trained model!!!")


	def print_train_info(self, epoch, mini_batch, cum_loss, k_print=10):
	  """
	  print some training info
	  """

	  # print loss
	  if mini_batch % k_print == k_print-1:

	    # print info
	    print('epoch: {}, mini-batch: {}, loss: [{:.5f}]'.format(epoch + 1, mini_batch + 1, cum_loss / k_print))

	    # zero cum loss
	    cum_loss = 0.0

	  return cum_loss

  def train(self):
  	"""
  	train interface
  	"""
  	pass


  def eval(self):
  	"""
  	evaluation interface
  	"""
  	pass



class CnnHandler(NetHandler):
	"""
	Neural Network Mentor for CNNs
	"""

  def __init__(self, nn_arch, batch_archiv):

		# parent class init
		super().__init__(nn_arch, batch_archiv)

		# loss criterion
	  self.criterion = torch.nn.CrossEntropyLoss()


  def train(self, num_epochs=2, lr=1e-3, momentum=0.5, param_str='nope'):
  	"""
  	train the neural network
  	"""

	  # create optimizer
	  #optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
	  optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

	  # collect losses over epochs
	  train_loss = np.zeros(num_epochs)
	  val_loss = np.zeros(num_epochs)
	  val_acc = np.zeros(num_epochs)

	  print("\n--Training starts:")

	  # start time
	  start_time = time.time()

	  # epochs
	  for epoch in range(num_epochs):

	    # cumulated loss
	    cum_loss = 0.0

	    # TODO: do this with loader function from pytorch (maybe or not)
	    # fetch data samples
	    for i, (x, y) in enumerate(zip(x_train, y_train)):

	      # zero parameter gradients
	      optimizer.zero_grad()

	      # forward pass o:[b x c]
	      o = self.model(x)

	      # loss
	      loss = self.criterion(o, y)

	      # backward
	      loss.backward()

	      # optimizer step - update params
	      optimizer.step()

	      # loss update
	      cum_loss += loss.item()

	      # batch loss
	      train_loss[epoch] += cum_loss

	      # print some infos, reset cum_loss
	      cum_loss = print_train_info(epoch, i, cum_loss, k_print=y_train.shape[0] // 10)

	    # valdiation
	    val_loss[epoch], val_acc[epoch], _ = self.eval(self.model, self.x_val, self.y_val, self.classes, logging_enabled=False)

	    # TODO: Early stopping if necessary
	    
	  print('--Training finished')

	  # log time
	  logging.info('Traning on arch: {}  time: {}'.format(param_str, s_to_hms_str(time.time() - start_time)))

	  return model, train_loss, val_loss, val_acc


	def eval(x_eval, y_eval, classes, z_eval=None, logging_enabled=True, calc_cm=False, verbose=False):
	  """
	  evaluation of nn
	  """

	  # init
	  correct, total, eval_loss, cm = 0, 0, 0.0, None
	  y_all, y_hat_all = np.empty(shape=(0), dtype=y_eval.numpy().dtype), np.empty(shape=(0), dtype=y_eval.numpy().dtype)

	  # no gradients for eval
	  with torch.no_grad():

	    # load data
	    for i, (x, y) in enumerate(zip(x_eval, y_eval)):

	      # classify
	      o = self.model(x)

	      # loss
	      eval_loss += self.criterion(o, y)

	      # prediction
	      _, y_hat = torch.max(o.data, 1)

	      # add total amount of prediction
	      total += y.size(0)

	      # check if correctly predicted
	      correct += (y_hat == y).sum().item()

	      # collect labels for confusion matrix
	      if calc_cm:
	        y_all = np.append(y_all, y)
	        y_hat_all = np.append(y_hat_all, y_hat)

	      # some prints
	      if verbose:
	        if z_eval is not None:
	          print("\nlabels: {}".format(z_eval[i]))
	        print("output: {}\npred: {}\nactu: {}, \t corr: {} ".format(o.data, y_hat, y, (y_hat == y).sum().item()))

	  # print accuracy
	  eval_log = "Eval: correct: [{} / {}] acc: [{:.4f}] with loss: [{:.4f}]\n".format(correct, total, 100 * correct / total, eval_loss)
	  print(eval_log)

	  # confusion matrix
	  if calc_cm:
	    cm = confusion_matrix(y_all, y_hat_all)

	  # log to file
	  if logging_enabled:
	    logging.info(eval_log)

	  return eval_loss, (correct / total), cm








