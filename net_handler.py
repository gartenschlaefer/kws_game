"""
Handling Neural Networks
"""

import numpy as np
import torch

import time

from conv_nets import ConvNetTrad, ConvNetFstride4
from adversarial_nets import G_experimental, D_experimental

from score import TrainScore, EvalScore

# TODO: Remove this later
# used for evaluation
import torchvision.utils as vutils


class NetHandler():
	"""
	Neural Network Handler with general functionalities and interfaces
	"""

	def __init__(self, nn_arch, use_cpu=False):

		# neural net architecture (see in config which are available : string)
		self.nn_arch = nn_arch

		# use cpu or gpu
		self.use_cpu = use_cpu

		# set device
		self.device = torch.device("cuda:0" if (torch.cuda.is_available() and not use_cpu) else "cpu")

		# print msg
		print("device: ", self.device)
		if torch.cuda.is_available() and not use_cpu:
			print("use gpu: ", torch.cuda.get_device_name(self.device))
		

	def get_nn_model(self):
		"""
		simply get the desired nn model
		"""

		# select network architecture
		if self.nn_arch == 'conv-trad':

			# traditional conv-net
			return ConvNetTrad(self.n_classes)

		elif self.nn_arch == 'conv-fstride':

			# limited multipliers conv-net
			return ConvNetFstride4(self.n_classes)

		elif self.nn_arch == 'adv-experimental':

			# adversarial network
			return (G_experimental(), D_experimental())

		# architecture not found
		print("***Network Architecture not found!")

		# traditional conv-net
		return None


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


	def load_model(self, path_coll, for_what='train'):
		"""
		loads model
		"""
		pass


	def save_model(self, path_coll, train_params, class_dict, train_score=None, save_as_pre_model=False):
		"""
		saves model
		"""
		pass


	def train_nn(self, train_params, batch_archive):
		"""
		train interface
		"""
		return TrainScore(train_params['num_epochs'])


	def eval_nn(self, eval_set, batch_archive, calc_cm=False, verbose=False):
		"""
		evaluation interface
		"""
		return EvalScore()


	def classify_sample(self):
		"""
		classify a single sample
		"""
		pass


	def generate_samples(self, noise=None, num_samples=10, to_np=False):
		"""
		generate samples if it is a generative network
		"""
		pass


	def set_eval_mode(self):
		"""
		sets the eval mode (dropout layers are ignored)
		"""
		pass



class CnnHandler(NetHandler):
	"""
	Neural Network Handler for CNNs
	"""

	def __init__(self, nn_arch, n_classes, use_cpu=False):

		# parent class init
		super().__init__(nn_arch, use_cpu)

		# classes
		self.n_classes = n_classes

		# loss criterion
		self.criterion = torch.nn.CrossEntropyLoss()

		# get the right nn model
		self.model = self.get_nn_model()

		# model to device
		try:
			self.model.to(self.device)
		except:
			print("***model could not be sent to device!")


	def load_model(self, path_coll, for_what='train'):
		"""
		load model
		"""

		# which model should be loaded
		if for_what == 'trained':
			model_file_path = path_coll.model_file

		elif for_what == 'pre':
			model_file_path = path_coll.model_pre_file

		elif for_what == 'classifier':
			model_file_path = path_coll.classifier_model

		# load model
		try:
			print("load model: ", model_file_path)
			self.model.load_state_dict(torch.load(model_file_path))

		except:
			print("\n***could not load pre-trained model!!!\n")


	def save_model(self, path_coll, train_params, class_dict, train_score=None, save_as_pre_model=False):
		"""
		saves model
		"""

		# just save the model
		torch.save(self.model.state_dict(), path_coll.model_file)

		# save param file
		np.savez(path_coll.params_file, nn_arch=self.nn_arch, train_params=train_params, class_dict=class_dict)

		# save metric file
		if train_score is not None:
			np.savez(path_coll.metrics_file, train_score=train_score)

		# save also als pre model
		if save_as_pre_model:
			torch.save(self.model.state_dict(), path_coll.model_pre_file)


	def train_nn(self, train_params, batch_archive):
		"""
		train the neural network
		train_params: {'num_epochs': [], 'lr': [], 'momentum': []}
		"""

		# create optimizer
		#optimizer = torch.optim.SGD(self.model.parameters(), lr=train_params['lr'], momentum=train_params['momentum'])
		optimizer = torch.optim.Adam(self.model.parameters(), lr=train_params['lr'])

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
				o = self.model(x)

				# loss
				loss = self.criterion(o, y)

				# backward
				loss.backward()

				# optimizer step - update params
				optimizer.step()

				# update batch loss collection
				train_score.update_batch_losses(epoch, loss.item())

				# print some infos
				self.print_train_info(epoch, i, train_score, k_print=batch_archive.y_train.shape[0] // 10)
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
				o = self.model(x)

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


	def eval_select_set(self, eval_set, batch_archive):
		"""
		select set to evaluate
		"""

		# select the set
		x_eval, y_eval, z_eval = None, None, None

		# validation set
		if eval_set == 'val':
			x_eval, y_eval, z_eval = batch_archive.x_val, batch_archive.y_val, None

		# test set
		elif eval_set == 'test':
			x_eval, y_eval, z_eval = batch_archive.x_test, batch_archive.y_test, None

		# my test set
		elif eval_set == 'my':
			x_eval, y_eval, z_eval = batch_archive.x_my, batch_archive.y_my, batch_archive.z_my

		# set not found
		else:
			print("wrong usage of eval_nn, select eval_set one out of ['val', 'test', 'my']")

		return x_eval, y_eval, z_eval


	def classify_sample(self, x):
		"""
		classification of a single sample
		"""

		# input to tensor
		x = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(x.astype(np.float32)), 0), 0).to(self.device)

		# no gradients for eval
		with torch.no_grad():

			# classify
			o = self.model(x)

			# prediction
			_, y_hat = torch.max(o.data, 1)

		return int(y_hat), o


	def set_eval_mode(self):
		"""
		sets the eval mode (dropout layers are ignored)
		"""
		self.model.eval()



class AdversarialNetHandler(NetHandler):
	"""
	Adversarial Neural Network Handler
	adapted form: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
	"""

	def __init__(self, nn_arch, use_cpu=False):

		# parent class init
		super().__init__(nn_arch, use_cpu)

		# loss criterion
		self.criterion = torch.nn.BCELoss()

		# neural network models, G-Generator, D-Discriminator
		self.G, self.D = self.get_nn_model()

		# model to device
		self.G.to(self.device)
		self.D.to(self.device)

		# labels
		self.real_label = 1.
		self.fake_label = 0.

		# weights init
		self.G.apply(self.weights_init)
		self.D.apply(self.weights_init)


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


	def load_model(self, path_coll, for_what='trained'):
		"""
		load model
		"""

		# which model should be loaded
		if for_what == 'trained':
			g_file = path_coll.adv_g_model_file
			d_file = path_coll.adv_d_model_file

		elif for_what == 'pre':
			g_file = None
			d_file = None

		elif for_what == 'classifier':
			g_file = None
			d_file = None

		else:
			print("for_what parameter was wrong defined")


		# load model
		try:
			print("load adv_g_model_file: ", g_file)
			print("load adv_d_model_file: ", d_file)
			self.G.load_state_dict(torch.load(g_file))
			self.D.load_state_dict(torch.load(d_file))

		except:
			print("\n***could not load pre-trained model!!!\n")


	def save_model(self, path_coll, train_params, class_dict, train_score=None, save_as_pre_model=False):
		"""
		saves model
		"""

		# just save the model
		torch.save(self.G.state_dict(), path_coll.adv_g_model_file)
		torch.save(self.D.state_dict(), path_coll.adv_d_model_file)

		# save param file
		np.savez(path_coll.params_file, nn_arch=self.nn_arch, train_params=train_params, class_dict=class_dict)

		# save metric file
		if train_score is not None:
			np.savez(path_coll.metrics_file, train_score=train_score)

		# TODO:
		# save also als pre model
		#if save_as_pre_model and model_pre_file is not None:
		#	torch.save(self.model.state_dict(), model_pre_file)


	def train_nn(self, train_params, batch_archive, callback_f=None):
		"""
		train adversarial nets
		"""

		# Create batch of latent vectors that we will use to visualize the progression of the generator
		fixed_noise = torch.randn(32, self.G.n_latent, device=self.device)

		# Setup Adam optimizers for both G and D
		optimizer_d = torch.optim.Adam(self.D.parameters(), lr=train_params['lr'], betas=(train_params['beta'], 0.999))
		optimizer_g = torch.optim.Adam(self.G.parameters(), lr=train_params['lr'], betas=(train_params['beta'], 0.999))

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
				self.D.zero_grad()
				optimizer_d.zero_grad()

				# --
				# train with real batch

				# labels for batch
				y = torch.full((batch_archive.batch_size,), self.real_label, dtype=torch.float, device=self.device)

				# forward pass o:[b x c]
				o = self.D(x).view(-1)

				# loss of D with reals
				d_loss_real = self.criterion(o, y)
				d_loss_real.backward()
				#d_reals = o.mean().item()


				# --
				# train with fake batch

				# create noise as input
				noise = torch.randn(batch_archive.batch_size, self.G.n_latent, device=self.device)

				# create fakes through Generator
				fakes = self.G(noise)

				# create fake labels
				y.fill_(self.fake_label)

				# fakes to D
				o = self.D(fakes.detach()).view(-1)

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

				self.G.zero_grad()

				y.fill_(self.real_label)

				o = self.D(fakes).view(-1)

				# loss of G of D with fakes
				g_loss = self.criterion(o, y)
				g_loss.backward()
				#D_G_z2 = o.mean().item()

				# optimizer step
				optimizer_g.step()

				# update batch loss collection
				train_score.update_batch_losses(epoch, loss=0.0, g_loss=g_loss.item(), d_loss_real=d_loss_real.item(), d_loss_fake=d_loss_fake.item())

				# print some infos
				self.print_train_info(epoch, i, train_score, k_print=batch_archive.y_train.shape[0] // 10)
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
			noise = torch.randn(num_samples, self.G.n_latent, device=self.device)

		# create fakes through Generator
		with torch.no_grad():
			fakes = torch.squeeze(self.G(noise).detach().cpu())

		# to numpy if necessary
		if to_np:
			fakes = fakes.numpy()

		return fakes



def cnn_analytics(cfg, batch_archive):
	"""
	evaluate convolutional networks
	"""

	# create an cnn handler
	cnn_handler = CnnHandler(nn_arch='conv-fstride', n_classes=5, use_cpu=False)

	# training
	cnn_handler.train_nn(cfg['ml']['train_params'], batch_archive=batch_archive)

	# validation
	cnn_handler.eval_nn(eval_set='val', batch_archive=batch_archive, calc_cm=False, verbose=False)

	# classify sample
	y_hat, o = cnn_handler.classify_sample(np.random.randn(39, 32))

	print("classify: [{}]\noutput: [{}]".format(y_hat, o))


def image_collect(x):
	"""
	collect images of mfcc's (used as callback function in the training of adversarial networks)
	"""

	# image list for evaluation
	global img_list

	# append image
	img_list.append(x)


def create_anim(path_coll):
	"""
	create image animation
	"""

	# image list for evaluation
	global img_list

	# images for evaluation on training
	print("amount of mfccs for anim: ", len(img_list))

	# plot
	fig = plt.figure(figsize=(8,8))
	#plt.axis("off")

	# animation
	ani = animation.ArtistAnimation(fig, [[plt.imshow(i[0, :], animated=True)] for i in img_list], interval=1000, repeat_delay=1000, blit=True)
	plt.show()

	# save
	ani.save("{}anim.mp4".format(path_coll.model_path))


def adversarial_analytics(cfg, path_coll, batch_archive):
	"""
	evaluate the adversarial networks
	"""

	# adversarial
	adv_handler = AdversarialNetHandler(nn_arch='adv-experimental', use_cpu=False)

	# reduce batch archieve to one label


	# check if model already exists
	if not os.path.exists(path_coll.adv_g_model_file) or not os.path.exists(path_coll.adv_d_model_file) or cfg['ml']['retrain']:

		# train
		train_score = adv_handler.train_nn(cfg['ml']['train_params'], batch_archive=batch_archive, callback_f=image_collect)

		# save model
		adv_handler.save_model(path_coll=path_coll, train_params=cfg['ml']['train_params'], class_dict=batch_archive.class_dict, train_score=train_score, save_as_pre_model=cfg['ml']['save_as_pre_model'])

		# plots
		plot_train_loss(train_score.train_loss, train_score.val_loss, plot_path=path_coll.model_path, name='train_loss')

	# load model params from file without training
	else:

		# load model
		adv_handler.load_model(path_coll=path_coll, for_what='trained')

	# generate samples from trained model
	fake = adv_handler.generate_samples(num_samples=1, to_np=True)
	plot_mfcc_only(fake, fs=16000, hop=160, plot_path=None, name='None')

	# training anim
	create_anim(path_coll)



if __name__ == '__main__':
	"""
	handles all neural networks with training, evaluation and classifying samples 
	"""

	import yaml
	import matplotlib.pyplot as plt
	import matplotlib.animation as animation
	import os

	from batch_archive import SpeechCommandsBatchArchive
	from path_collector import PathCollector
	from plots import plot_val_acc, plot_train_loss, plot_mfcc_only

	# yaml config file
	cfg = yaml.safe_load(open("./config.yaml"))

	# path collector
	path_coll = PathCollector(cfg)

	# create all necessary folders
	path_coll.create_ml_folders()

	# create batches
	batch_archive = SpeechCommandsBatchArchive(path_coll.mfcc_data_files_all, batch_size=32, batch_size_eval=4)

	# global vars
	global img_list
	img_list = []

	# cnn analytics
	#cnn_analytics(cfg, batch_archive)

	# adversarial analytics
	#adversarial_analytics(cfg, path_coll, batch_archive)
