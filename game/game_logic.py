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

    # add input handler
    self.input_handler = InputKeyHandler(self)


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


  def esc_key(self):
    """
    if esc is pressed
    """

    # end loop
    self.run_loop = False


  def event_update(self, event):
    """
    update game logic
    """

    # quit game
    if event.type == pygame.QUIT: self.run_loop, self.quit_game = False, True
    
    # input handling
    self.input_handler.handle(event)



class ThingsGameLogic(GameLogic):
  """
  Game Logic for things.py
  """

  def __init__(self, level, levels):

    # parent init
    super().__init__()

    # vars
    self.level = level
    self.levels = levels

    # level id always start with zero
    self.level_id = 0


  def check_win_condition(self):
    """
    move character to position
    """

    # henry found the things
    if self.level.interactable_dict['henry'].things_collected if 'henry' in self.level.interactable_dict.keys() else False:

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

      # update level id
      self.level_id += 1

      # clamp level id
      if self.level_id >= len(self.levels): self.level_id = 0

      # restart
      self.restart_game()

      # new level
      self.level = self.levels[self.level_id]


  def update(self):
    """
    update game logic
    """

    # check win condition
    self.check_win_condition()

    return self.level



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