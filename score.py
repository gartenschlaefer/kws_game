"""
Score classes for training and evaluation
"""

import numpy as np


class TrainScore():
	"""
	collection of scores
	"""

	def __init__(self, num_epochs, use_adv=False):

		# params
		self.num_epochs = num_epochs
		self.use_adv = use_adv

		# collect losses over epochs
		self.batch_loss = 0.0
		self.train_loss = np.zeros(self.num_epochs)
		self.val_loss = np.zeros(self.num_epochs)
		self.val_acc = np.zeros(self.num_epochs)

		# adversarial losses
		self.g_batch_loss = None
		self.d_batch_loss_real = None
		self.d_batch_loss_fake = None
		self.g_train_loss = None
		self.d_train_loss_real = None
		self.d_train_loss_fake = None

		# init if activated
		if self.use_adv:
			self.g_loss = np.zeros(self.num_epochs)
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
		if self.use_adv:
			self.g_batch_loss = 0.0
			self.d_batch_loss_real = 0.0
			self.d_batch_loss_fake = 0.0


	def update_batch_losses(self, epoch, loss, g_loss=None, d_loss_real=None, d_loss_fake=None):
		"""
		update losses
		"""

		self.batch_loss = loss
		self.train_loss += loss 

		if self.use_adv:

			# batch loss for score
			self.g_batch_loss = g_loss
			self.d_batch_loss_real = d_loss_real
			self.d_batch_loss_fake = d_loss_fake

			self.g_loss[epoch] += g_loss
			self.d_loss_real[epoch] += d_loss_real
			self.d_loss_fake[epoch] += d_loss_fake



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
		self.acc = 0.0
		self.cm = None

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

		# accuracy
		self.acc = 100 * self.correct / self.total

		# print eval results
		self.info_log(do_print=True)

		# no cm was required
		if not self.calc_cm:
			return

		# confusion matrix
		self.cm = confusion_matrix(self.y_all, self.y_hat_all)


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
