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
    self.won_game = False

    # callbacks
    self.win_condition_callback = self.complete_win_condition if True else self.no_win_condition
    self.loose_condition_callback = self.no_loose_condition


  def check_win_condition(self):
    """
    win condition callback
    """
    self.win_condition_callback()


  def check_loose_condition(self):
    """
    loose condition callback
    """
    self.loose_condition_callback()


  def no_win_condition(self):
    """
    no win condition call
    """
    pass


  def complete_win_condition(self):
    """
    no win condition call
    """
    if self.complete: self.end_game, self.run_loop, self.won_game = True, False, True


  def no_loose_condition(self):
    """
    no win condition call
    """
    pass


  def esc_key(self):
    """
    if esc is pressed
    """

    # end loop
    self.run_loop = False
    self.complete = True
    self.won_game = True


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
    self.check_loose_condition()


  def reset(self):
    """
    reset
    """

    # reset vars
    self.end_game = False
    self.run_loop = True
    self.quit_game = False
    self.complete = False
    self.won_game = False



class ThingsGameLogic(GameLogic):
  """
  Game Logic for things.py
  """

  def __init__(self, level):

    # parent init
    super().__init__()

    # vars
    self.level = level

    # win condition with character
    self.win_condition_callback = self.character_win_condition if 'character' in self.level.interactable_dict.keys() else self.no_win_condition
    self.loose_condition_callback = self.enemy_loose_condition if 'enemy' in self.level.interactable_dict.keys() else self.no_loose_condition


  def character_win_condition(self):
    """
    characters win conditions
    """

    # check if things are collected
    if self.level.interactable_dict['character'].things_collected and not self.end_game:

      # win the level
      self.level.win()

      # end the game
      self.end_game = True

      # won the game flag
      self.won_game = True


  def enemy_loose_condition(self):
    """
    enemy loose condition
    """

    if self.level.interactable_dict['enemy'].hit_enemy:

      # win the level
      self.level.loose()

      # end the game
      self.end_game = True

      # loose the game flag
      self.won_game = False


  def enter_key(self):
    """
    if enter key is pressed
    """
    
    # end of game reached, ask for enter
    if self.end_game:

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