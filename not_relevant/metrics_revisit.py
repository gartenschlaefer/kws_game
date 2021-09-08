"""
revisit metrics
"""

import numpy as np

from pathlib import Path
from glob import glob

import sys
sys.path.append("../")

from audio_dataset import AudioDataset, SpeechCommandsDataset, MyRecordingsDataset
from batch_archive import SpeechCommandsBatchArchive
from test_bench import TestBench
from legacy import legacy_adjustments_net_params
from plots import plot_train_score, plot_confusion_matrix, plot_mfcc_only, plot_grid_images, plot_mfcc_anim


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


  def run_test_bench(self):
    """
    test bench run
    """

    # shift invariance test
    self.test_bench.test_invariances()


  def run_score(self):
    """
    score plots
    """

    # load metrics
    metrics = np.load(self.metrics_file, allow_pickle=True)

    # see whats in data
    print(metrics.files)

    train_score_dict = metrics['train_score_dict'][()]

    print("train_score_dict: ", train_score_dict.keys())
    print("score_class: ", train_score_dict['score_class'])

    # plot train score
    plot_train_score(train_score_dict, plot_path=self.model_path, name_ext='_revisit', show_plot=False)


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

          # plot images
          plot_grid_images(x, context=context, color_balance=True, padding=1, num_cols=np.clip(x.shape[0], 1, 8), plot_path=self.model_path, name=k)
          plot_grid_images(x, context=context+'-div', color_balance=True, padding=1, num_cols=np.clip(x.shape[0], 1, 8), plot_path=self.model_path, name='div_' + k)

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

    # score print of collected
    #eval_score.info_collected(self.net_handler.nn_arch, self.audio_dataset.param_path, self.train_params, info_file=self.score_file, do_print=False)

    # print confusion matrix
    print("confusion matrix:\n{}\n".format(eval_score.cm))

    # plot confusion matrix
    plot_confusion_matrix(eval_score.cm, self.batch_archive.class_dict.keys(), plot_path=self.model_path, name='confusion_test')


    # --
    # evaluation on my set

    # check if my set exists
    if not 'my' in self.batch_archive.set_names: return

    print("\n--Evaluation on My Set:")

    # evaluation of model
    eval_score = self.net_handler.eval_nn('my', batch_archive=self.batch_archive, collect_things=True, verbose=True)
    
    # score print of collected
    #eval_score.info_collected(self.net_handler.nn_arch, self.audio_dataset.param_path, self.train_params, info_file=self.score_file, do_print=False)

    # confusion matrix
    print("confusion matrix:\n{}\n".format(eval_score.cm))

    # plot confusion matrix
    plot_confusion_matrix(eval_score.cm, self.batch_archive.class_dict.keys(), plot_path=self.model_path, name='confusion_my')



if __name__ == '__main__':
  """
  main
  """

  import yaml

  # yaml config file
  cfg = yaml.safe_load(open("../config.yaml"))

  # metric path
  #model_path = '../ignore/models/hyb-jim/v5_c7n1m1_n-500_r1-5_mfcc32-12_c1d0d0e0_norm1_f-1x12x50/bs-32_it-1000_lr-d-0p0001_lr-g-0p0001/'
  model_path = '../docu/best_models/exp_cepstral/'

  # select models
  model_sel = ['conv-fstride', 'conv-jim', 'conv-trad']

  # model dictionary
  model_path_dict = {'{}'.format(ms): [{'model_path': str(m).split('cnn_model.pth')[0], 'model': str(m), 'params': str(p), 'metrics': str(me)} for i, (m, p, me) in enumerate(zip(Path(model_path).rglob('cnn_model.pth'), Path(model_path).rglob('params.npz'), Path(model_path).rglob('metrics.npz'))) if str(m).find(ms) != -1] for ms in model_sel}


  print("model: ", model_path_dict['conv-fstride'][0])
  #stop

  # metrics revisit
  metrics_revisit = MetricsRevisit(cfg, model_path_dict=model_path_dict['conv-fstride'][0])

  # metrics test bench
  #metrics_revisit.run_test_bench()
  #metrics_revisit.run_score()
  #metrics_revisit.run_weights()
  metrics_revisit.run_eval()

