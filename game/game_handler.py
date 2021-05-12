"""
game handler
"""

import pygame

from color_bag import ColorBag
from interactable import Interactable
from game_logic import GameLogic
from menu import Menu


class GameHandler():
  """
  game handler
  """

  def __init__(self, cfg):

    # arguments
    self.cfg = cfg

    # init pygame
    pygame.init()

    # init display
    self.screen = pygame.display.set_mode(cfg['game']['screen_size'])

    # menu
    self.menu = Menu(self.cfg['game'], self.screen)

    # options menu

    # levels

    # game loop
    self.game_loop_state_dict = {'main_menu_loop': 0, 'option_menu_loop': 1, 'game_loop': 2, 'exit': 3}

    # start at main menu
    self.game_loop_state = 'main_menu_loop'


  def run_game(self):
    """
    run game loops
    """

    while self.game_loop_state != 'exit':

      # run menu loop
      if self.game_loop_state == 'main_menu_loop': self.game_loop_state = self.menu.menu_loop()

      # run options menu loop
      #elif self.game_loop_state == 'option_menu_loop': new_loop = self.menu.menu_loop()
      #elif self.game_loop_state == 'game_loop': new_loop = self.menu.menu_loop()
      else: self.game_loop_state = 'exit'


if __name__ == '__main__':
  """
  main
  """

  import yaml

  # yaml config file
  cfg = yaml.safe_load(open("../config.yaml"))


  # game handler
  game_handler = GameHandler(cfg)

  # run game
  game_handler.run_game()

  # end pygame
  pygame.quit()