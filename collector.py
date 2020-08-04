"""
class for collectors
"""

import numpy as np
import queue


class Collector:
  """
  collector class for audio processing
  """

  def __init__(self, N=400, hop=160, frame_size=32, update_size=2):
    """
    constructor
    """

    # determine data size
    data_size = int((frame_size * hop + N ) /  hop)

    print("Collector init with datasize: ", data_size)

    # flags
    self.is_collecting = False

    # data containers
    self.qu = queue.Queue(maxsize=data_size)
    self.x = np.empty(shape=(0), dtype=np.float32)

    # size of pre frames for update
    self.update_size = update_size


  def start_collecting(self):
    """
    start collecting data e.g from onset
    """

    self.is_collecting = True


  def update_collect(self, x):
    """
    update the collection with an entry
    """

    # usual update of new item
    if not self.is_collecting:

      # remove last
      if self.qu.qsize() == self.update_size:
        dummy = self.qu.get_nowait()

      # error message
      if self.qu.qsize() > self.update_size:
        print("wrong handling of buffer")

    # put conditions
    if not self.qu.full():
      self.qu.put(x)


  def read_collection(self):
    """
    read out whole collection
    """

    self.is_collecting = False

    # read out elements
    while not self.qu.empty():
      self.x = np.concatenate((self.x, self.qu.get_nowait()))

    y = self.x.copy()

    # reset x
    self.x = np.empty(shape=(0), dtype=np.float32)

    return y


  def is_full(self):
    """
    get full info
    """

    return self.qu.full()