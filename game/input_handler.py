"""
input handler class
"""

import pygame
from interactable import Interactable


class InputHandler(Interactable):
  """
  Input Handler class
  """

  def __init__(self, objs):

    # object to be handled
    self.objs = objs



class InputKeyHandler(InputHandler):
  """
  Keyboard handler
  """

  def event_update(self, event):
    """
    handle keyboard inputs
    """

    # key down
    if event.type == pygame.KEYDOWN:

      # movement      
      if event.key == pygame.K_LEFT or event.key == pygame.K_a: [obj.direction_change([-1, 0]) for obj in self.objs if obj.is_moveable()]
      elif event.key == pygame.K_RIGHT or event.key == pygame.K_d: [obj.direction_change([1, 0]) for obj in self.objs if obj.is_moveable()]
      if event.key == pygame.K_UP or event.key == pygame.K_w: [obj.direction_change([0, -1]) for obj in self.objs if obj.is_moveable()]
      elif event.key == pygame.K_DOWN or event.key == pygame.K_s: [obj.direction_change([0, 1]) for obj in self.objs if obj.is_moveable()]

      # other buttons
      if event.key == pygame.K_SPACE: [obj.action_key() for obj in self.objs]
      if event.key == pygame.K_RETURN: [obj.enter_key() for obj in self.objs]
      if event.key == pygame.K_ESCAPE: [obj.esc_key() for obj in self.objs]
      if event.key == pygame.K_r: [obj.r_key() for obj in self.objs]
      if event.key == pygame.K_n: [obj.n_key() for obj in self.objs]

    # key up
    elif event.type == pygame.KEYUP:

      #if not obj.grid_move:
      if event.key == pygame.K_LEFT or event.key == pygame.K_a: [obj.direction_change([1, 0]) for obj in self.objs if obj.is_moveable() and not obj.grid_move]
      elif event.key == pygame.K_RIGHT or event.key == pygame.K_d: [obj.direction_change([-1, 0]) for obj in self.objs if obj.is_moveable() and not obj.grid_move]
      if event.key == pygame.K_UP or event.key == pygame.K_w: [obj.direction_change([0, 1]) for obj in self.objs if obj.is_moveable() and not obj.grid_move]
      elif event.key == pygame.K_DOWN or event.key == pygame.K_s: [obj.direction_change([0, -1]) for obj in self.objs if obj.is_moveable() and not obj.grid_move]



class InputMicHandler(InputHandler):
  """
  Microphone handler
  """

  def __init__(self, objs, mic):

    # init of father class
    super().__init__(objs)

    # arguments
    self.mic = mic


  def update(self):
    """
    handle mic inputs
    """

    # get command
    command = self.mic.update_read_command()

    # interpret command
    if command is not None: [(obj.speech_command(command), print("obj: {}, mic command: {}".format(obj.__class__.__name__, command))) for obj in self.objs]