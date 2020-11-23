"""
class for collectors
"""

import numpy as np
import queue


class Collector:
  """
  collector class for audio processing
  """

  def __init__(self, N=400, hop=160, frame_size=32, update_size=10, frames_post=10, is_audio_record=False):
    """
    constructor
    """

    # determine data size
    self.data_size = int((frame_size * hop + N ) /  hop) + frames_post

    # params
    self.update_size = update_size
    self.frames_post = frames_post
    self.is_audio_record = is_audio_record

    # data containers
    self.qu = queue.Queue(maxsize=self.data_size)
    self.x = np.empty(shape=(0), dtype=np.float32)

    # whole audio data
    self.x_all = np.empty(shape=(0), dtype=np.float32)

    # vars
    self.is_collecting = False
    self.collection_counter = 0


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

    # save all
    if self.is_audio_record:
      self.x_all = np.concatenate((self.x_all, x))
      

  def read_collection(self):
    """
    read out whole collection
    """

    self.is_collecting = False

    # reset x
    self.x = np.empty(shape=(0), dtype=np.float32)

    # read out elements
    while not self.qu.empty():
      self.x = np.concatenate((self.x, self.qu.get_nowait()))

    # update collection counter
    self.collection_counter += 1

    return self.x


  def is_full(self):
    """
    get full info
    """

    return self.qu.full()