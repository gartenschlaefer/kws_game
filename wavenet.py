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

  def __init__(self, in_channels, out_channels, skip_channels=16, dilation=1):

    # parent init
    super().__init__()

    # arguments
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.skip_channels = skip_channels
    self.dilation = dilation

    # dilated convolution filter and gate
    self.conv_filter = nn.Conv1d(self.in_channels, self.out_channels, kernel_size=2, stride=1, padding=0, dilation=self.dilation, groups=1, bias=False, padding_mode='zeros')
    self.conv_gate = nn.Conv1d(self.in_channels, self.out_channels, kernel_size=2, stride=1, padding=0, dilation=self.dilation, groups=1, bias=False, padding_mode='zeros')

    # 1 x 1 convolution for skip connection
    self.conv_skip = nn.Conv1d(self.out_channels, out_channels=self.skip_channels, kernel_size=2, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros')
    

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

    # padding for skip connection
    o = torch.nn.functional.pad(o, (1, 0), 'constant', 0)

    # skip connection
    skip = self.conv_skip(o)
    o = self.conv_skip(o)

    # add residual
    o += residual

    return o, skip



class Wavenet(nn.Module):
  """
  wavenet class
  """

  def __init__(self):

    # parent init
    super().__init__()

    self.in_channels = 1
    self.out_channels = 16
    self.skip_channels = 16

    # n classes
    self.n_classes = 256

    # layers of wavenet blocks -> receptive field: 2^n_layers
    self.n_layers = 8

    # wavenet_layers
    self.wavenet_layers = torch.nn.ModuleList()

    # first block
    self.wavenet_layers.append(WavenetResBlock(self.in_channels, self.out_channels, skip_channels=self.skip_channels, dilation=1))

    # append further blocks
    for i in range(1, self.n_layers): self.wavenet_layers.append(WavenetResBlock(self.out_channels, self.out_channels, skip_channels=self.skip_channels, dilation=2**i))

    # conv layer post
    self.conv_post1 = nn.Conv1d(in_channels=self.skip_channels, out_channels=16, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros')
    self.conv_post2 = nn.Conv1d(in_channels=16, out_channels=self.n_classes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros')


  def forward(self, x):
    """
    forward connection
    """

    # init output and skips
    o, skips = x, torch.zeros(x.shape[0], self.skip_channels, x.shape[-1])

    # wavenet layers
    for wavenet_layer in self.wavenet_layers: 

      # wavenet layer
      o, skip = wavenet_layer(o)

      # sum skips
      skips += skip

    # relu of summed skip
    x = torch.relu(skips)

    # conv layers post
    x = torch.relu(self.conv_post1(x))
    x = self.conv_post2(x)

    return x


  def mu_softmax(self, x, mu=256):
    """
    mu softmax function
    """
    return np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + np.ones(x.shape) * mu)


  def quantize(self, x, n_classes=256):
    """
    quantize
    """
    return torch.from_numpy(np.digitize(self.mu_softmax(x, mu=n_classes), bins=np.linspace(-1, 1, n_classes)) - 1).detach()


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
