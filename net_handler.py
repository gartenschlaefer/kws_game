"""
Handling Neural Networks
"""

import numpy as np
import torch

from conv_nets import *


class NetHandler():
	"""
	Neural Network Handler
	"""

	def __init__(self, nn_arch, batch_archiv):

		# neural net architecture (see in config which are available : string)
		self.nn_arch = nn_arch

		# training set
		self.batch_archiv = batch_archiv

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


	def load_model(self, model_path):
	  """
	  load model
	  """

	  # same model
	  if model_path is None:
	    return

	  # load model
	  try:
	    print("load model: ", model_path)
	    self.model.load_state_dict(torch.load(model_path))

	  except:
	    print("could not load pre-trained model!!!")


	def save_model(self, model_path):
		"""
		saves model
		"""

		# just save the model
		torch.save(self.model.state_dict(), model_path)


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


	def train_nn(self):
		"""
		train interface
		"""
		pass


	def eval_nn(self):
		"""
		evaluation interface
		"""
		pass



class CnnHandler(NetHandler):
	"""
	Neural Network Mentor for CNNs
	"""

	def __init__(self, nn_arch, batch_archiv, pre_trained_model_path=None):

		# parent class init
		super().__init__(nn_arch, batch_archiv, pre_trained_model_path=None)

		# loss criterion
		self.criterion = torch.nn.CrossEntropyLoss()


	def train_nn(self, num_epochs=2, lr=1e-3, momentum=0.5, param_str='nope'):
		"""
		train the neural network
		"""

		# create optimizer
		#optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
		optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

		# score collector
		score_collector = ScoreCollecor(num_epochs)

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
		    score_collector.train_loss[epoch] += cum_loss

		    # print some infos, reset cum_loss
		    cum_loss = print_train_info(epoch, i, cum_loss, k_print=y_train.shape[0] // 10)

		  # valdiation
		  eval_score = self.eval_nn('val')

		  # update score collector
		  score_collector.val_loss[epoch], score_collector.val_acc[epoch] = eval_score.loss, eval_score.acc

		  # TODO: Early stopping if necessary
		  
		print('--Training finished')

		# log time
		score_collector.time_usage = time.time() - start_time 

		return score_collector


	def eval_nn(eval_set, calc_cm=False, verbose=False):
	  """
	  evaluation of nn
	  use eval_set out of ['val', 'test', 'my']
	  """

	  # init score
	  eval_score = EvalScore(label_dtype=batch_archiv.y_eval.numpy().dtype, calc_cm=calc_cm)

	  # select the evaluation set
	  x_eval, y_eval, z_eval = eval_select_set(eval_set)

	  # no gradients for eval
	  with torch.no_grad():

	    # load data
	    for i, (x, y) in enumerate(zip(x_eval, y_eval)):

	      # classify
	      o = self.model(x)

	      # loss
	      loss = self.criterion(o, y)

	      # prediction
	      _, y_hat = torch.max(o.data, 1)

	      # update eval score
	      eval_score.update(loss, y, y_hat)

	      # some prints
	      if verbose:
	        if z_eval is not None:
	          print("\nlabels: {}".format(z_eval[i]))
	        print("output: {}\npred: {}\nactu: {}, \t corr: {} ".format(o.data, y_hat, y, (y_hat == y).sum().item()))

	  # finish up scores
	  eval_score.finish()

	  return eval_score


	def eval_select_set(self, eval_set):
		"""
		select set to evaluate
		"""

		# select the set
		x_eval, y_eval, z_eval = None, None, None

		if eval_set == 'val':
			x_eval, y_eval, z_eval = batch_archiv.x_val, batch_archiv.y_val, None

		elif eval_set == 'test':
			x_eval, y_eval, z_eval = batch_archiv.x_test, batch_archiv.y_test, None

		# determine evaluation files
		elif eval_set == 'my':
			x_eval, y_eval, z_eval = batch_archiv.x_my, batch_archiv.y_my, batch_archiv.z_my

		else:
			print("wrong usage of eval_nn, select eval_set one out of ['val', 'test', 'my']")

		return x_eval, y_eval, z_eval



class ScoreCollecor():
	"""
	collection of scores
	"""

	def __init__(self, num_epochs):

		# collect losses over epochs
		self.train_loss = np.zeros(num_epochs)
		self.val_loss = np.zeros(num_epochs)
		self.val_acc = np.zeros(num_epochs)

		# time score
		self.time_usage = None



class EvalScore():
	"""
	typical scores for evaluation
	"""

	def __init__(self, label_dtype, calc_cm=False):

		# params
		self.label_dtype = label_dtype
		self.calc_cm = calc_cm

		# init vars
		self.loss = 0.0
		self.correct = 0
		self.total = 0
		self.cm = None
		self.acc = None

		# all labels from batches
		self.y_all = np.empty(shape=(0), dtype=self.label_dtype)

		# all predicted labels
		self.y_hat_all = np.empty(shape=(0), dtype=self.label_dtype)


	def update(self, loss, y, y_hat):
		"""
		update score (batch update)
		"""

		# loss update
		self.loss += loss

		# add total amount of prediction
		self.total += y.size(0)

		# check if correctly predicted
		self.correct += (y_hat == y).sum().item()

		# collect labels for confusion matrix
		if self.calc_cm:
			self.y_all = np.append(self.y_all, y)
			self.y_hat_all = np.append(self.y_hat_all, y_hat)


	def finish(self):
		"""
		finishing procedure
		"""

		from sklearn.metrics import confusion_matrix

		# no cm was required
		if not self.calc_cm:
			return

		# confusion matrix
		self.cm = confusion_matrix(self.y_all, self.y_hat_all)

		# accuracy
		self.acc = 100 * self.correct / self.total


	def info_log(self, do_print=True):
		"""
		print evaluation outcome
		"""

		# message
		eval_log = "Eval: correct: [{} / {}] acc: [{:.4f}] with loss: [{:.4f}]\n".format(self.correct, self.total, self.acc, self.loss)

		# print
		if do_print:
			print(eval_log)

		return eval_log






if __name__ == '__main__':
	"""
		handles all neural networks with training and evaluation 
	"""

	import yaml
	from batch_archiv import BatchArchiv

	# yaml config file
	cfg = yaml.safe_load(open("./config.yaml"))

	# create batches
	batch_archiv = BatchArchiv(cfg['ml']['mfcc_data_files'], batch_size=32, batch_size_eval=4)

	# select architecture
	nn_arch = 'conv-fstride'

	# create an cnn handler
	cnn_handler = CnnHandler(nn_arch, batch_archiv)

	# training
	cnn_handler.train_nn(num_epochs=2, lr=1e-3, momentum=0.5, param_str='nope')

	# validation
	cnn_handler.eval_nn(calc_cm=False, verbose=False, use_my_eval=False)
