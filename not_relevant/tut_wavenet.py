"""
wavenets from master oord
"""

import torch
import torch.nn as nn


class WavenetResBlock(nn.Module):
  """
  wavenet residual block
  """

  def __init__(self, in_channels, out_channels, dilation=1):

    # parent init
    super().__init__()

    # arguments
    self.in_channels = in_channels
    self.out_channels = out_channels

    # dilated convolution filter and gate
    self.conv_filter = nn.Conv1d(self.in_channels, self.out_channels, kernel_size=2, stride=1, padding=1, dilation=dilation, groups=1, bias=False, padding_mode='zeros')
    self.conv_gate = nn.Conv1d(self.in_channels, self.out_channels, kernel_size=2, stride=1, padding=1, dilation=dilation, groups=1, bias=False, padding_mode='zeros')

    # 1 x 1 convolution for skip connection
    self.conv_skip = nn.Conv1d(self.out_channels, out_channels=self.in_channels, kernel_size=2, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros')
    

  def forward(self, x):
    """
    forward connection
    """

    print("x: ", x.shape)

    # residual connection
    residual = x

    # 1. conv layers
    o_f = torch.tanh(self.conv_filter(x))
    o_g = torch.sigmoid(self.conv_gate(x))

    # element-wise multiplication
    o = o_f * o_g
    print("o1: ", o.shape)

    # skip connection
    skip = self.conv_skip(o)
    o = self.conv_skip(o)

    print("o2: ", o.shape)
    print("r: ", residual.shape)

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

    # wavenet blocks
    self.wavenet_block1 = WavenetResBlock(in_channels=1, out_channels=16)

    # conv layer post
    self.conv_post1 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros')
    self.conv_post2 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros')


  def mu_softmax(self, x, mu=256):
    """
    mu softmax function
    """

    return torch.sign(x) * torch.log(1 + mu * torch.abs(x)) / torch.log(1 + mu)


  def forward(self, x):
    """
    forward connection
    """

    # 1. wavenet block
    o, skip = self.wavenet_block1(x)

    # relu of summed skip
    x = torch.relu(skip)

    # conv layers post
    x = torch.relu(self.conv_post1(x))
    x = self.conv_post2(x)

    # softmax
    x = self.mu_softmax(x, mu=256)

    return x




if __name__ == '__main__':
  """
  main
  """

  # input
  x = torch.randn(1, 1, 2)

  # wavenet
  wavenet = Wavenet()

  # next sample
  y = wavenet(x)

  print("wavenet: ", wavenet)