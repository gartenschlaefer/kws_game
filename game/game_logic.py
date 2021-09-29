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

    # variables
    self.end_game = False
    self.run_loop = True
    self.quit_game = False
    self.complete = False


  def check_win_condition(self):
    """
    move character to position
    """
    if self.complete: self.end_game, self.run_loop = True, False


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


  def esc_key(self):
    """
    if esc is pressed
    """

    # end loop
    self.run_loop = False
    self.complete = True


  def event_update(self, event):
    """
    update game logic
    """

    # quit game
    if event.type == pygame.QUIT: self.run_loop, self.quit_game = False, True


  def update(self):
    """
    update game logic
    """

    # check win condition
    self.check_win_condition()


  def reset(self):
    """
    reset
    """

    # reset vars
    self.end_game = False
    self.run_loop = True
    self.quit_game = False
    self.complete = False



class ThingsGameLogic(GameLogic):
  """
  Game Logic for things.py
  """

  def __init__(self, level):

    # parent init
    super().__init__()

    # vars
    self.level = level


  def check_win_condition(self):
    """
    move character to position
    """

    # character found the things
    if self.level.interactable_dict['character'].things_collected if 'character' in self.level.interactable_dict.keys() else False:

      # win the level
      self.level.win()

      # end the game
      self.end_game = True


  def restart_game(self):
    """
    restart game
    """
    
    # reset end game logic
    self.end_game = False

    # reset game objects
    self.level.reset()


  def enter_key(self):
    """
    if enter key is pressed
    """
    
    if self.end_game:

      # restart
      self.restart_game()

      # level complete
      self.complete = True



class MenuGameLogic(GameLogic):
  """
  Game Logic for menues
  """

  def __init__(self, menu):

    # parent init
    super().__init__()

    # arguments
    self.menu = menu

    # exit with escape key
    self.esc_key_exit = False

    # for input handler
    self.grid_move = False


  def is_moveable(self):
    """
    moveable flag
    """
    return True


  def direction_change(self, direction):
    """
    arrow keys pressed
    """
    self.menu.direction_change(direction)


  def enter_key(self):
    """
    if action key is pressed
    """
    self.menu.enter_key()


  def esc_key(self):
    """
    if esc is pressed
    """
    self.menu.esc_key()


  def reset(self):
    """
    reset
    """

    # reset vars
    self.run_loop, self.esc_key_exit = True, False