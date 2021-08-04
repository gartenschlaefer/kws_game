"""
oh no this project need legacy adjustments
"""


def legacy_model_file_naming(f_model_name_dict):
  """
  changes the model file naming (key is changed) to actual standard, e.g. model was named merely as 'model.pth' instead of 'cnn_model.pth'
  """
  return dict((k, v) if k != '' else ('cnn', v) for k, v in f_model_name_dict.items()) if '' in f_model_name_dict.keys() else f_model_name_dict


def legacy_adjustments_feature_params(feature_params):
  """
  feature params legacy
  """

  # raw extension
  if 'use_mfcc_features' not in feature_params.keys(): feature_params.update({'use_mfcc_features': True})
  if 'frame_size_s' not in feature_params.keys(): feature_params.update({'frame_size_s': 0.5})

  # mfcc selection extension
  if 'use_channels' not in feature_params.keys(): feature_params.update({'use_channels': False})
  if 'use_cepstral_features' not in feature_params.keys(): feature_params.update({'use_cepstral_features': True})
  if 'use_delta_features' not in feature_params.keys(): feature_params.update({'use_delta_features': True})
  if 'use_double_delta_features' not in feature_params.keys(): feature_params.update({'use_double_delta_features': True})
  if 'use_energy_features' not in feature_params.keys(): feature_params.update({'use_energy_features': True})

  # old mfcc stuff, replace with new ones
  if 'compute_deltas' in feature_params.keys(): feature_params.update({'use_delta_features': feature_params['compute_deltas'], 'use_double_delta_features': feature_params['compute_deltas']})

  return feature_params


def legacy_adjustments_net_params(net_params):
  """
  net params for saved parameters, e.g. in classifier
  """

  # for legacy models
  try:
    data_size = net_params['data_size'][()]
  except:
    data_size = (1, 39, 32)
    print("old classifier model use fixed data size: {}".format(data_size))

  try:
    feature_params = net_params['feature_params'][()]
  except:
    feature_params = {'use_mfcc_features': True, 'fs': 16000, 'N_s': 0.025, 'hop_s': 0.010, 'frame_size_s': 0.32, 'n_filter_bands': 32, 'n_ceps_coeff': 12, 'frame_size': 32, 'norm_features': False, 'use_channels': False, 'use_cepstral_features': True, 'use_delta_features': True, 'use_double_delta_features': True, 'use_energy_features': True}
    print("old classifier model use fixed feature parameters: {}".format(feature_params))

  # feature params adjustment for old saves
  feature_params = legacy_adjustments_feature_params(feature_params)

  return data_size, feature_params



# --
# comments
#
# renames:
# energy_thres: 0.0001 -> energy_thresh: 0.0001
# changed energy_thresh to energy_thresh_db