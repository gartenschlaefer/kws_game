"""
color bag class
"""

import pygame


class ColorBag():
  """
  Input Handler class
  """

  def __init__(self):

    # background
    self.background = (255, 255, 255)

    # ordinary walls
    self.wall = (10, 200, 200)

    # moving walls
    self.active_move_wall = (200, 100, 100)
    self.default_move_wall = (10, 100, 100)

    # text color
    self.win = (50, 100, 100)


