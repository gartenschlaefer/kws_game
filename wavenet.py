"""
wavenets
some code snippets adapted from https://github.com/vincentherrmann/pytorch-wavenet
"""

import numpy as np
import torch
import torch.nn as nn

from conv_nets import ConvBasics


class WavenetResBlock(nn.Module):
  """
  wavenet residual block
  """

  def __init__(self, in_channels, out_channels, dilated_channels=16, pred_channels=64, dilation=1, n_samples=8000):

    # parent init
    super().__init__()

    # arguments
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.dilated_channels = dilated_channels
    self.pred_channels = pred_channels
    self.dilation = dilation
    self.n_samples = n_samples

    # average pooling
    self.av_pool_size = 160
    self.av_pool_stride = 80

    # sub sampling
    self.av_pool_sub = int(self.n_samples / self.av_pool_stride - 1)

    # use bias
    self.use_conv_bias = False

    # dilated convolution filter and gate
    self.conv_filter = nn.Conv1d(self.in_channels, self.dilated_channels, kernel_size=2, stride=1, padding=0, dilation=self.dilation, groups=1, bias=False, padding_mode='zeros')
    self.conv_gate = nn.Conv1d(self.in_channels, self.dilated_channels, kernel_size=2, stride=1, padding=0, dilation=self.dilation, groups=1, bias=False, padding_mode='zeros')

    # 1 x 1 convolution for skip connection
    self.conv_skip = nn.Conv1d(self.dilated_channels, out_channels=self.out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=self.use_conv_bias, padding_mode='zeros')
    
    # average pooling for class prediction, downsample to 10ms
    #self.av_pool = nn.AvgPool1d(kernel_size=self.av_pool_size, stride=self.av_pool_stride)
    self.av_pool = nn.Conv1d(self.dilated_channels, out_channels=self.dilated_channels, kernel_size=160, stride=80, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros')

    # conv prediction
    self.conv_pred1 = nn.Conv1d(self.dilated_channels, out_channels=self.pred_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=self.use_conv_bias, padding_mode='zeros')


  def forward(self, x):
    """
    forward connection
    """

    # input length
    input_len = x.shape[-1]

    # residual connection
    residual = torch.clone(x)

    # zero padding from left
    x = torch.nn.functional.pad(x, (self.dilation, 0), 'constant', 0)

    # 1. conv layers
    o_f = torch.tanh(self.conv_filter(x))
    o_g = torch.sigmoid(self.conv_gate(x))

    # element-wise multiplication
    o = o_f * o_g

    # average pooling for prediction
    y = self.av_pool(o)

    # prediction layer
    y = self.conv_pred1(y)

    # output and skip connection
    #skip = self.conv_skip(o)
    o = self.conv_skip(o)
    skip = torch.clone(o)

    # add residual
    o += residual

    return o, skip, y


  def count_params(self):
    """
    count all parameters
    """
    return [p.numel() for p in self.parameters() if p.requires_grad]


  def calc_amount_of_operations(self):
    """
    calculate amount of operations
    """

    # amount of ops
    n_ops = {
      'conv_filter': (self.in_channels * self.dilated_channels) * self.n_samples * (2 * 2 + 1), 
      'conv_gate': (self.in_channels * self.dilated_channels) * self.n_samples * (2 * 2 + 1),
      'conv_skip': (self.dilated_channels * self.out_channels) * self.n_samples * (2 * 1 + 1),
      'conv_pred': (self.dilated_channels * self.pred_channels) * self.av_pool_sub * (2 * 1 + 1)
      }

    return n_ops


class Wavenet(nn.Module, ConvBasics):
  """
  wavenet class
  """

  def __init__(self, n_classes, n_samples=8000):

    # parent init
    super().__init__()

    # arguments
    self.n_classes = n_classes
    self.n_samples = n_samples

    # channel params
    self.in_channels = 1
    self.out_channels = 1
    self.dilated_channels = 16
    self.skip_channels = 2
    self.pred_channels = 128

    # average pooling
    self.av_pool_size = 160
    self.av_pool_stride = 80

    # sub sampling
    self.av_pool_sub = int(self.n_samples / self.av_pool_stride - 1)

    # use bias
    self.use_conv_bias = True

    # n classes
    self.target_quant_size = 256

    # layers of wavenet blocks -> receptive field: 2^n_layers
    self.n_layers = 7

    # wavenet_layers
    self.wavenet_layers = torch.nn.ModuleList()

    # first block
    self.wavenet_layers.append(WavenetResBlock(self.in_channels, self.out_channels, dilated_channels=self.dilated_channels, pred_channels=self.pred_channels, dilation=1))

    # append further blocks
    for i in range(1, self.n_layers): self.wavenet_layers.append(WavenetResBlock(self.out_channels, self.out_channels, dilated_channels=self.dilated_channels, pred_channels=self.pred_channels, dilation=2**i))

    # conv layer post for skip connection
    self.conv_skip1 = nn.Conv1d(in_channels=self.out_channels, out_channels=self.skip_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=self.use_conv_bias, padding_mode='zeros')
    self.conv_skip2 = nn.Conv1d(in_channels=self.skip_channels, out_channels=self.target_quant_size, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=self.use_conv_bias, padding_mode='zeros')

    # conv layer predictions
    self.kernel_sizes_pred = [(1,), (1,)]
    self.strides_pred = [(1,), (1,)]
    self.padding_pred = [(0,), (0,)]

    # conv dimensions of predictions
    self.conv_dim_pred = self.get_conv_layer_dimensions((self.av_pool_sub,), self.kernel_sizes_pred, self.strides_pred, self.padding_pred)

    # conv layer post for class prediction
    self.conv_pred1 = nn.Conv1d(in_channels=self.pred_channels, out_channels=self.pred_channels, kernel_size=self.kernel_sizes_pred[0], stride=self.strides_pred[0], padding=self.padding_pred[0], dilation=1, groups=1, bias=self.use_conv_bias, padding_mode='zeros')
    self.conv_pred2 = nn.Conv1d(in_channels=self.pred_channels, out_channels=1, kernel_size=self.kernel_sizes_pred[1], stride=self.strides_pred[1], padding=self.padding_pred[1], dilation=1, groups=1, bias=self.use_conv_bias, padding_mode='zeros')

    # fc layer
    self.fc1 = nn.Linear(np.prod(self.conv_dim_pred[-1]), self.n_classes)

    # softmax layer
    self.softmax = nn.Softmax(dim=1)


  def forward(self, x):
    """
    forward connection
    """

    # output, skip, class
    o, s, y = self.wavenet_layers[0](x)

    # wavenet layers
    for i, wavenet_layer in enumerate(self.wavenet_layers[1:]): 

      # wavenet layer
      o, s_i, y_i = wavenet_layer(o)

      # sum skips
      s += s_i

      # sum predictions
      y += y_i

    # relu of summed skip
    t = torch.relu(s)

    # conv layers post
    t = torch.relu(self.conv_skip1(t))
    t = self.conv_skip2(t)

    # relu for prediction
    y = torch.relu(y)

    # conv layer prediction 1
    y = torch.relu(self.conv_pred1(y))

    # conv layer prediction 2
    y = torch.relu(self.conv_pred2(y))

    # flatten output from conv layer
    y = y.view(-1, np.product(y.shape[1:]))

    # fc layer and softmax
    y = self.softmax(self.fc1(y))

    return t, y


  def mu_softmax(self, x, mu=256):
    """
    mu softmax function
    """
    return np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + np.ones(x.shape) * mu)


  def quantize(self, x, quant_size=256):
    """
    quantize
    """
    a = self.mu_softmax(x.detach().cpu(), mu=quant_size)
    b = np.digitize(a, bins=np.linspace(-1, 1, quant_size)) - 1
    c = torch.from_numpy(b).detach().to(device=x.device)
    return c


  def count_params(self):
    """
    count parameters
    """
    return [p.numel() for p in self.parameters() if p.requires_grad]


  def calc_amount_of_operations(self):
    """
    calculate amount of operations
    """

    # resnet blocks
    n_ops = {'block{}'.format(i): sum(l.calc_amount_of_operations().values()) for i, l in enumerate(self.wavenet_layers)}

    # conv layers
    n_ops.update({
      'conv_skip1': (self.out_channels * self.skip_channels) * self.n_samples * (2 * 1 + 1),
      'conv_skip2': (self.skip_channels * self.target_quant_size) * self.n_samples * (2 * 1 + 1),
      'conv_pred1': (self.pred_channels * self.pred_channels) * self.av_pool_sub * (2 * 1 + 1),
      'conv_pred2': (self.pred_channels * 1) * self.av_pool_sub * (2 * 1 + 1)
      })

    return n_ops



if __name__ == '__main__':
  """
  main
  """

  # resnet block
  res_block = WavenetResBlock(1, 1, dilated_channels=16, pred_channels=64, dilation=1)

  # wavenet
  wavenet = Wavenet(n_classes=5)

  # next sample
  y = wavenet(torch.randn(1, 1, 8000))

  # prints
  #print("wavenet: ", wavenet)
  print("wavenet layer params: {}".format(wavenet.count_params()))

  # operations
  print("res block ops: ", res_block.calc_amount_of_operations())
  print("res block params: {}, total: {}".format(res_block.count_params(), sum(res_block.count_params())))
  print("\nwavenet ops: ", wavenet.calc_amount_of_operations())
  print("\nwavenet params: ", wavenet.count_params())
  print("total number of params: {}, total number of ops: {:,}".format(sum(wavenet.count_params()), sum(wavenet.calc_amount_of_operations().values())))
