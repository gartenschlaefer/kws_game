"""
wavenets
some code snippets adapted from https://github.com/vincentherrmann/pytorch-wavenet
"""

import numpy as np
import torch
import torch.nn as nn


class WavenetResBlock(nn.Module):
  """
  wavenet residual block
  """

  def __init__(self, in_channels, out_channels, skip_channels=16, pred_channels=16, dilation=1):

    # parent init
    super().__init__()

    # arguments
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.skip_channels = skip_channels
    self.pred_channels = pred_channels
    self.dilation = dilation

    # dilated convolution filter and gate
    self.conv_filter = nn.Conv1d(self.in_channels, self.out_channels, kernel_size=2, stride=1, padding=0, dilation=self.dilation, groups=1, bias=False, padding_mode='zeros')
    self.conv_gate = nn.Conv1d(self.in_channels, self.out_channels, kernel_size=2, stride=1, padding=0, dilation=self.dilation, groups=1, bias=False, padding_mode='zeros')

    # 1 x 1 convolution for skip connection
    self.conv_skip = nn.Conv1d(self.out_channels, out_channels=self.skip_channels, kernel_size=2, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros')
    
    # average pooling for class prediction, downsample to 10ms
    self.av_pool = nn.AvgPool1d(kernel_size=160, stride=80)

    # conv prediction 1
    self.conv_pred1 = nn.Conv1d(self.out_channels, out_channels=self.pred_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros')


  def forward(self, x):
    """
    forward connection
    """

    #print("x: ", x.shape)

    # input length
    input_len = x.shape[-1]

    # residual connection
    residual = x

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

    # padding for skip connection
    o = torch.nn.functional.pad(o, (1, 0), 'constant', 0)

    # skip connection
    skip = self.conv_skip(o)
    o = self.conv_skip(o)

    # add residual
    o += residual

    return o, skip, y



class Wavenet(nn.Module):
  """
  wavenet class
  """

  def __init__(self, n_classes):

    # parent init
    super().__init__()

    # arguments
    self.n_classes = n_classes

    # channel params
    self.in_channels = 1
    self.out_channels = 16
    self.skip_channels = 16
    self.pred_channels = 16

    # n classes
    self.target_quant_size = 256

    # layers of wavenet blocks -> receptive field: 2^n_layers
    self.n_layers = 8

    # wavenet_layers
    self.wavenet_layers = torch.nn.ModuleList()

    # first block
    self.wavenet_layers.append(WavenetResBlock(self.in_channels, self.out_channels, skip_channels=self.skip_channels, pred_channels=self.pred_channels, dilation=1))

    # append further blocks
    for i in range(1, self.n_layers): self.wavenet_layers.append(WavenetResBlock(self.out_channels, self.out_channels, skip_channels=self.skip_channels, pred_channels=self.pred_channels, dilation=2**i))

    # conv layer post for skip connection
    self.conv_skip1 = nn.Conv1d(in_channels=self.skip_channels, out_channels=16, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros')
    self.conv_skip2 = nn.Conv1d(in_channels=16, out_channels=self.target_quant_size, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros')

    # conv layer post for class prediction
    self.conv_pred1 = nn.Conv1d(in_channels=self.pred_channels, out_channels=1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros')

    # fc layer
    self.fc1 = nn.Linear(99, self.n_classes)


  def forward(self, x):
    """
    forward connection
    """

    # init output and skips
    o, skips, y = x, torch.zeros(x.shape[0], self.skip_channels, x.shape[-1]).to(device=x.device), torch.zeros(x.shape[0], self.pred_channels, 99).to(device=x.device)

    # wavenet layers
    for wavenet_layer in self.wavenet_layers: 

      # wavenet layer
      o, skip, y = wavenet_layer(o)

      # sum skips
      skips += skip

      # sum predictions
      y += y

    # relu of summed skip
    t = torch.relu(skips)

    # conv layers post
    t = torch.relu(self.conv_skip1(t))
    t = self.conv_skip2(t)

    # relu for prediction
    y = torch.relu(y)

    # conv layer prediction
    y = self.conv_pred1(y)

    # another relu
    y = torch.relu(y)

    # flatten output from conv layer
    y = y.view(-1, np.product(y.shape[1:]))

    # fc layer
    y = self.fc1(y)

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



if __name__ == '__main__':
  """
  main
  """

  # wavenet
  wavenet = Wavenet()

  # next sample
  y = wavenet(torch.randn(1, 1, 16000))

  # count params
  layer_params = wavenet.count_params()

  # prints
  print("wavenet: ", wavenet)
  print("layer: {} sum: {}".format(layer_params, sum(layer_params)))
