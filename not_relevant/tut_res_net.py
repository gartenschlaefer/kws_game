"""
Residual network tutorial from https://pytorch-tutorial.readthedocs.io/en/latest/tutorial/chapter03_intermediate/3_2_2_cnn_resnet_cifar10/
"""

# ---------------------------------------------------------------------------- #
# An implementation of https://arxiv.org/pdf/1512.03385.pdf                    #
# See section 4.2 for the model architecture on CIFAR-10                       #
# Some part of the code was referenced from below                              #
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py   #
# ---------------------------------------------------------------------------- #

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


class ResidualBlock(nn.Module):
  """
  res block
  """

  def __init__(self, in_channels, out_channels, stride=1, downsample=None):

    # init
    super(ResidualBlock, self).__init__()

    # conv layer
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)

    # conv layer
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.downsample = downsample


  def forward(self, x):
    """
    forward connection
    """

    # save res state
    residual = x

    # res block
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)

    # downsample
    if self.downsample: residual = self.downsample(x)

    # add res
    out += residual

    # last output
    out = self.relu(out)

    return out



class ResNet(nn.Module):
  """
  resnet
  """

  def __init__(self, block, layers, num_classes=10):

    # init parent
    super(ResNet, self).__init__()

    # in channels
    self.in_channels = 16

    # conv layer
    self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn = nn.BatchNorm2d(16)
    self.relu = nn.ReLU(inplace=True)

    # res blocks as layers
    self.layer1 = self.make_layer(block, 16, layers[0])
    self.layer2 = self.make_layer(block, 32, layers[1], 2)
    self.layer3 = self.make_layer(block, 64, layers[2], 2)

    # average pooling
    self.avg_pool = nn.AvgPool2d(8)

    # fully connected
    self.fc = nn.Linear(64, num_classes)


  def make_layer(self, block, out_channels, blocks, stride=1):
    """
    make res blocks, blocks is number of blocks
    """

    # no downsampling
    downsample = None

    if (stride != 1) or (self.in_channels != out_channels):
        downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(out_channels))
    
    # layers
    layers = []
    layers.append(block(self.in_channels, out_channels, stride, downsample))

    self.in_channels = out_channels

    # append block to layer
    for i in range(1, blocks): layers.append(block(out_channels, out_channels))

    # create sequential
    return nn.Sequential(*layers)


  def forward(self, x):
    """
    forward connection
    """

    # first conv
    out = self.conv(x)
    out = self.bn(out)
    out = self.relu(out)

    # res blocks
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)

    # average pooling
    out = self.avg_pool(out)
    out = out.view(out.size(0), -1)

    # last output
    out = self.fc(out)
    return out




def update_lr(optimizer, lr):
  """
  update learning rate
  """

  # update learning rate
  for param_group in optimizer.param_groups: param_group['lr'] = lr



if __name__ == '__main__':
  """
  main
  """

  # --
  # dataset

  # Device configuration
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Hyper-parameters
  num_epochs = 80
  learning_rate = 0.001

  # Image preprocessing modules
  transform = transforms.Compose([transforms.Pad(4), transforms.RandomHorizontalFlip(), transforms.RandomCrop(32), transforms.ToTensor()])

  # CIFAR-10 dataset
  train_dataset = torchvision.datasets.CIFAR10(root='./ignore/cifar10', train=True, transform=transform, download=True)
  test_dataset = torchvision.datasets.CIFAR10(root='./ignore/cifar10', train=False, transform=transforms.ToTensor())

  # loaders
  train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
  test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)


  # --
  # net

  # init resnet
  model = ResNet(ResidualBlock, [2, 2, 2]).to(device)

  # Loss and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


  # Train the model
  total_step = len(train_loader)
  curr_lr = learning_rate

  # epochs
  for epoch in range(num_epochs):

    # train loader
    for i, (images, labels) in enumerate(train_loader):

      # input
      images, labels = images.to(device), labels.to(device)

      # Forward pass
      outputs = model(images)
      loss = criterion(outputs, labels)

      # Backward and optimize
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # info
      if (i+1) % 100 == 0: print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Decay learning rate
    if (epoch+1) % 20 == 0:
      curr_lr /= 3
      update_lr(optimizer, curr_lr)


  # Test the model
  model.eval()

  with torch.no_grad():

    # init
    correct, total = 0, 0

    # test loader
    for images, labels in test_loader:

      # input
      images, labels = images.to(device), labels.to(device)

      # output
      outputs = model(images)

      # prediction
      _, predicted = torch.max(outputs.data, 1)

      # scores
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

    # info
    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

  # Save the model checkpoint
  #torch.save(model.state_dict(), 'resnet.ckpt')