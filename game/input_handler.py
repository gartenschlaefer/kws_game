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

    for obj in self.objs:

      # key down
      if event.type == pygame.KEYDOWN:

        if obj.is_moveable():
          if event.key == pygame.K_LEFT or event.key == pygame.K_a: obj.direction_change([-1, 0])
          elif event.key == pygame.K_RIGHT or event.key == pygame.K_d: obj.direction_change([1, 0])
          if event.key == pygame.K_UP or event.key == pygame.K_w: obj.direction_change([0, -1])
          elif event.key == pygame.K_DOWN or event.key == pygame.K_s: obj.direction_change([0, 1])

        # jump button
        if event.key == pygame.K_SPACE: obj.action_key()
        if event.key == pygame.K_RETURN: obj.enter_key()
        if event.key == pygame.K_ESCAPE: obj.esc_key()

      # key up
      elif event.type == pygame.KEYUP and obj.is_moveable():

        if not obj.grid_move:
          if event.key == pygame.K_LEFT or event.key == pygame.K_a: obj.direction_change([1, 0])
          elif event.key == pygame.K_RIGHT or event.key == pygame.K_d: obj.direction_change([-1, 0])
          if event.key == pygame.K_UP or event.key == pygame.K_w: obj.direction_change([0, 1])
          elif event.key == pygame.K_DOWN or event.key == pygame.K_s: obj.direction_change([0, -1])



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
    handle keyboard inputs
    """

    # get command
    command = self.mic.update_read_command()

    # interpret command
    if command is not None:

      for obj in self.objs:

        print("obj: {}, mic command: {}".format(obj.__class__.__name__, command))
        obj.speech_command(command)

        # # direction
        # if command == 'left': obj.direction_change([-1, 0])
        # elif command == 'right': obj.direction_change([1, 0])
        # elif command == 'up': obj.direction_change([0, -1])
        # elif command == 'down': obj.direction_change([0, 1])

        # # action
        # elif command == 'go': obj.action_key()

        # # other actions
        # elif command == '_noise' or command == '_mixed': 
        #   print("yeah")
        #   obj.actions(action=command)
