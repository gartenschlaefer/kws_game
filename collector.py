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

    # arguments
    self.N = N
    self.hop = hop
    self.frame_size = frame_size
    self.update_size = update_size
    self.frames_post = frames_post
    self.is_audio_record = is_audio_record

    # determine data size
    self.data_size = int((frame_size * hop + N ) /  hop) + frames_post

    # data containers
    self.q = queue.Queue(maxsize=self.data_size)
    self.x = np.empty(shape=(0), dtype=np.float32)

    # whole audio data for recording
    self.x_all = np.empty(shape=(0), dtype=np.float32)
    self.e_all = np.empty(shape=(0), dtype=np.float32)
    self.on_all = np.empty(shape=(0), dtype=np.float32)

    # vars
    self.is_collecting = False
    self.collection_counter = 0


  def start_collecting(self):
    """
    start collecting data e.g from onset
    """

    self.is_collecting = True


  def update_collect(self, x, e=None, on=None):
    """
    update the collection with an entry
    """

    # usual update of new item
    if not self.is_collecting:

      # remove last
      if self.q.qsize() == self.update_size:
        dummy = self.q.get_nowait()

      # error message
      if self.q.qsize() > self.update_size:
        print("wrong handling of buffer")

    # put conditions
    if not self.q.full():
      self.q.put(x)

    # save all for audio record
    if self.is_audio_record:
      self.x_all = np.append(self.x_all, x)
      self.e_all = np.append(self.e_all, e)
      if on:
        self.on_all = np.append(self.on_all, len(self.x_all) / self.hop)


  def reset_collection_all(self):
    """
    reset the collection of all data during audio record
    """

    self.x_all = np.empty(shape=(0), dtype=np.float32)
    self.e_all = np.empty(shape=(0), dtype=np.float32)
    self.on_all = np.empty(shape=(0), dtype=np.float32)

      

  def read_collection(self):
    """
    read out whole collection
    """

    self.is_collecting = False

    # reset x
    self.x = np.empty(shape=(0), dtype=np.float32)

    # read out elements
    while not self.q.empty():
      self.x = np.append(self.x, self.q.get_nowait())

    # update collection counter
    self.collection_counter += 1

    return self.x


  def is_full(self):
    """
    get full info
    """

    return self.q.full()