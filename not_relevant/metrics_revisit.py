"""
revisit metrics
"""

import torch
import numpy as np
import re

from pathlib import Path
from glob import glob

import sys
sys.path.append("../")

from audio_dataset import AudioDataset, SpeechCommandsDataset, MyRecordingsDataset
from batch_archive import SpeechCommandsBatchArchive
from test_bench import TestBench
from legacy import legacy_adjustments_net_params
from plots import plot_train_score, plot_confusion_matrix, plot_mfcc_only, plot_grid_images, plot_mfcc_anim, plot_val_acc_multiple


class MetricsRevisit():
  """
  revisit metrics
  """

  def __init__(self, cfg, model_path_dict, root_path='./'):

    # arguments
    self.cfg = cfg
    self.model_path_dict = model_path_dict
    self.root_path = root_path

    # variables
    self.model_path = self.model_path_dict['model_path']
    self.model_file = self.model_path_dict['model']
    self.params_file = self.model_path_dict['params']
    self.metrics_file = self.model_path_dict['metrics']

    print("model_path_dict: ", model_path_dict)

    # test bench
    self.test_bench = TestBench(self.cfg['test_bench'], test_model_path=self.model_path, root_path='../')

    # net handler
    self.net_handler = self.test_bench.net_handler
    self.feature_params = self.test_bench.feature_params

    # show plot
    self.show_plot = True

    # param dict
    self.param_dict = {
      'mfcc': re.sub(r'([_])|(32-)', '', re.findall(r'_mfcc[0-9]+-[0-9]+_', self.model_path)[0]),
      'norm': re.sub('norm', '', re.findall(r'norm[01]', self.model_path)[0]),
      'feature_sel': re.sub(r'[_]', '', re.findall(r'_c[01]d[01]d[01]e[01]_', self.model_path)[0]),
      }

    print("params: ", self.param_dict)


  def get_cepstral_labels(self, underline=False):
    """
    get labels for mfcc norm
    """
    return '{} norm{}'.format(self.param_dict['mfcc'], self.param_dict['norm']) if not underline else '{}_norm{}'.format(self.param_dict['mfcc'], self.param_dict['norm'])


  def get_mfcc_labels(self, underline=False):
    """
    get labels for mfcc norm
    """
    return '{}'.format(self.param_dict['feature_sel'])


  def run_test_bench(self, plot_path=None, name_pre='', name_post=''):
    """
    test bench run
    """

    # plot path
    plot_path = self.model_path if plot_path is None else plot_path

    # shift invariance test
    self.test_bench.test_invariances(plot_path=plot_path, name_pre=name_pre, name_post=name_post, plot_prob=False)


  def run_score(self, use_average=False):
    """
    score plots
    """

    # train score dict
    train_score_dict = self.get_train_score_dict(average_acc=use_average)

    # plot train score
    plot_train_score(train_score_dict, plot_path=self.model_path, name_ext='_revisit', show_plot=self.show_plot)


  def get_train_score_dict(self, average_acc=False):
    """
    get train score dict
    """

    # load metrics
    metrics = np.load(self.metrics_file, allow_pickle=True)

    # see whats in data
    print(metrics.files)

    # get train score
    train_score_dict = metrics['train_score_dict'][()]

    if average_acc:

      # average pool
      av_pool = torch.nn.AvgPool1d(kernel_size=10, stride=1)

      # adapt train score
      train_score_dict['val_acc'] = np.squeeze(av_pool(torch.unsqueeze(torch.unsqueeze(torch.from_numpy(train_score_dict['val_acc']), dim=0), dim=0)).numpy())

    return train_score_dict


  def run_weights(self):
    """
    weights
    """

    # analyze weights
    for model_name, model in self.net_handler.models.items():

      # go through each weight
      for k, v in model.state_dict().items():

        # convolutional layers
        if k.find('conv') != -1 and not k.__contains__('bias'):

          # context for color map
          if k.find('layers.0.weight') != -1: context = 'weight0'
          elif k.find('layers.1.weight') != -1: context = 'weight1'
          else: context = 'weight0'

          # detach weights
          x = v.detach().cpu().numpy()

          # number of columns
          #num_cols = np.clip(x.shape[0], 1, 8)
          num_cols = 3

          # plot images
          plot_grid_images(x, context=context, color_balance=True, padding=1, num_cols=num_cols, plot_path=self.model_path, name=k, show_plot=self.show_plot)
          plot_grid_images(x, context=context+'-div', color_balance=True, padding=1, num_cols=num_cols, plot_path=self.model_path, name='div_' + k)

    # generate samples from trained model (only for adversarial)
    fakes = self.net_handler.generate_samples(num_samples=30, to_np=True)
    if fakes is not None:
      plot_grid_images(x=fakes, context='mfcc', padding=1, num_cols=5, plot_path=self.model_path, name='generated_samples_grid')
      plot_mfcc_only(fakes[0, 0], fs=16000, hop=160, plot_path=self.model_path, name='generated_sample')


  def run_eval(self):
    """
    evaluation
    """

    # audio sets
    self.audio_dataset = AudioDataset(self.cfg['datasets']['speech_commands'], self.feature_params, root_path='../')
    self.audio_dataset_my = AudioDataset(self.cfg['datasets']['my_recordings'], self.feature_params, root_path='../')

    # create batch archive
    self.batch_archive = SpeechCommandsBatchArchive(feature_file_dict={**self.audio_dataset.feature_file_dict, **self.audio_dataset_my.feature_file_dict}, batch_size_dict={'train': 32, 'test': 5, 'validation': 5, 'my': 1}, shuffle=True) if self.audio_dataset_my is not None else SpeechCommandsBatchArchive(feature_file_dict=self.audio_dataset.feature_file_dict, batch_size_dict={'train': 32, 'test': 5, 'validation': 5, 'my': 1}, shuffle=True)

    # create batches
    self.batch_archive.create_batches()

    print(self.batch_archive.x_set_dict.keys())
    #stop
    print("\n--Evaluation on Test Set:")

    # evaluation of model
    eval_score = self.net_handler.eval_nn('test', batch_archive=self.batch_archive, collect_things=True, verbose=False)

    # print confusion matrix
    print("confusion matrix:\n{}\n".format(eval_score.cm))

    # plot confusion matrix
    plot_confusion_matrix(eval_score.cm, self.batch_archive.class_dict.keys(), plot_path=self.model_path, name='confusion_test', show_plot=self.show_plot)


    # --
    # evaluation on my set

    # check if my set exists
    if not 'my' in self.batch_archive.set_names: return

    print("\n--Evaluation on My Set:")

    # evaluation of model
    eval_score = self.net_handler.eval_nn('my', batch_archive=self.batch_archive, collect_things=True, verbose=True)
    
    # confusion matrix
    print("confusion matrix:\n{}\n".format(eval_score.cm))

    # plot confusion matrix
    plot_confusion_matrix(eval_score.cm, self.batch_archive.class_dict.keys(), plot_path=self.model_path, name='confusion_my', show_plot=self.show_plot)


  def run_all(self):
    """
    run all
    """

    self.show_plot = False
    self.run_test_bench()
    self.run_score()
    self.run_weights()
    self.run_eval()



class MetricsCollector():
  """
  collects metric revisiter
  """

  def __init__(self, cfg, model_path, model_sel):

    # arguments
    self.cfg = cfg
    self.model_path = model_path
    self.model_sel = model_sel

    # model dictionary
    self.model_path_dict = {'{}'.format(ms): [{'model_path': str(m).split('cnn_model.pth')[0], 'model': str(m), 'params': str(p), 'metrics': str(me)} for i, (m, p, me) in enumerate(zip(Path(model_path).rglob('cnn_model.pth'), Path(model_path).rglob('params.npz'), Path(model_path).rglob('metrics.npz'))) if str(m).find(ms) != -1] for ms in model_sel}

    # metric revisiters
    self.metrics_revisit_dict = {model: [MetricsRevisit(self.cfg, model_path_dict=m) for m in self.model_path_dict[model]] for model in self.model_sel}


  def run_all_metrics(self):
    """
    run all metrics
    """
    [[mr.run_all() for mr in self.metrics_revisit_dict[model]] for model in self.model_sel]


  def accuracy_plot_cepstral(self, plot_path):
    """
    accuracy plot
    """

    for model in self.model_sel:

      # create acc dicts
      val_accs_dict = {mr.get_cepstral_labels(): mr.get_train_score_dict(average_acc=True)['val_acc'] for mr in self.metrics_revisit_dict[model]}

      # plot all
      plot_val_acc_multiple(val_accs_dict, plot_path=plot_path, name='exp_fs_mfcc_acc_{}'.format(model), show_plot=True, close_plot=True)


  def accuracy_plot_mfcc(self, plot_path):
    """
    accuracy plot
    """

    for model in self.model_sel:

      # create acc dicts
      val_accs_dict = {mr.get_mfcc_labels(): mr.get_train_score_dict(average_acc=True)['val_acc'] for mr in self.metrics_revisit_dict[model]}

      # plot all
      plot_val_acc_multiple(val_accs_dict, plot_path=plot_path, name='exp_fs_mfcc_acc_{}'.format(model), show_plot=True, close_plot=True)


  def test_bench_plot_mfcc(self, plot_path):
    """
    test bench for mfcc feature selection
    """
    [[mr.run_test_bench(plot_path=plot_path, name_pre='exp_fs_mfcc_', name_post='_{}_{}'.format(model, mr.get_mfcc_labels())) for mr in self.metrics_revisit_dict[model]] for model in self.model_sel]


if __name__ == '__main__':
  """
  main
  """

  import yaml

  # yaml config file
  cfg = yaml.safe_load(open("../config.yaml"))

  # plot path
  plot_path = '../docu/thesis/5_exp/figs/'

  # select models
  #model_sel = ['conv-fstride', 'conv-jim', 'conv-trad']

  # model dictionary
  #model_path_dict = {'{}'.format(ms): [{'model_path': str(m).split('cnn_model.pth')[0], 'model': str(m), 'params': str(p), 'metrics': str(me)} for i, (m, p, me) in enumerate(zip(Path(model_path).rglob('cnn_model.pth'), Path(model_path).rglob('params.npz'), Path(model_path).rglob('metrics.npz'))) if str(m).find(ms) != -1] for ms in model_sel}

  #print("model path: ", model_path_dict)
  #stop
  # metrics revisit
  #metrics_revisit = MetricsRevisit(cfg, model_path_dict=model_path_dict['conv-fstride'][0])

  # metrics test bench
  #metrics_revisit.run_test_bench()
  #metrics_revisit.run_score()
  #metrics_revisit.run_weights()
  #metrics_revisit.run_eval()
  #metrics_revisit.run_all()


  # cepstral
  #metrics_collector = MetricsCollector(cfg=cfg, model_path='../docu/best_models/ignore/exp_cepstral/', model_sel=['conv-fstride', 'conv-jim', 'conv-trad'])

  # accuracy plot
  #metrics_collector.accuracy_plot_cepstral(plot_path=plot_path, name='exp_fs_cepstral_acc_{}'.format(model))

  # run all metrics
  #metrics_collector.run_all_metrics()


  # mfcc
  metrics_collector = MetricsCollector(cfg=cfg, model_path='../docu/best_models/ignore/exp_mfcc/', model_sel=['conv-jim'])

  # run all metrics
  #metrics_collector.run_all_metrics()

  # accuracy plot
  #metrics_collector.accuracy_plot_mfcc(plot_path=plot_path)
  metrics_collector.test_bench_plot_mfcc(plot_path=plot_path)

