"""
Score classes for training and evaluation
"""

import numpy as np


class TrainScore():
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
