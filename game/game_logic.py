"""
game logic
"""

import pygame

from interactable import Interactable
from input_handler import InputKeyHandler


class GameLogic(Interactable):
  """
  Game Logic class interface
  """

  def __init__(self):

    self.end_game = False
    self.run_loop = True


  def check_win_condition(self):
    """
    move character to position
    """
    pass


  def check_loose_condition(self):
    """
    check loose condition
    """
    pass


  def restart_game(self):
    """
    restart game
    """
    pass


  def update(self):
    """
    update game logic
    """
    pass


  def event_update(self, event):
    """
    update game logic
    """

    # quit game
    if event.type == pygame.QUIT: 
      self.run_loop = False

    # end game
    elif event.type == pygame.KEYDOWN:
      if event.key == pygame.K_ESCAPE:
        self.run_loop = False



class ThingsGameLogic(GameLogic):
  """
  Game Logic for things.py
  """

  def __init__(self, henry, text):

    # parent init
    super().__init__()

    # add input handler
    self.input_handler = InputKeyHandler(self)

    # game objects
    self.henry = henry
    self.text = text


  def check_win_condition(self):
    """
    move character to position
    """

    # henry found the things
    if self.henry.things_collected:

      # stop henry
      self.henry.is_active = False

      # write win
      self.text.win_message(big_pos=(275, 75), small_pos=(250, 125))

      self.end_game = True


  def restart_game(self):
    """
    restart game
    """
    
    # reset end game logic
    self.end_game = False

    # reset game objects
    self.text.reset_messages()
    self.henry.reset()


  def enter_key(self):
    """
    if enter key is pressed
    """
    
    if self.end_game:
      self.restart_game()


  def esc_key(self):
    """
    if esc is pressed
    """

    # end loop
    self.run_loop = False


  def update(self):
    """
    update game logic
    """

    # check win condition
    self.check_win_condition()


  def event_update(self, event):
    """
    update game logic
    """
    
    # input handling
    self.input_handler.handle(event)