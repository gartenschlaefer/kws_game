"""
game handler
"""

import pygame

from color_bag import ColorBag
from interactable import Interactable
from game_logic import GameLogic
from menu import MainMenu, HelpMenu, OptionMenu


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

    # menues
    self.main_menu = MainMenu(self.cfg['game'], self.screen)
    self.help_menu = HelpMenu(self.cfg['game'], self.screen)
    self.option_menu = OptionMenu(self.cfg['game'], self.screen)

    # levels
    # todo...

    # game loop
    self.game_loop_state_dict = {'main_menu_loop': 0, 'game_loop': 1, 'help_menu_loop': 2, 'option_menu_loop': 3, 'exit': 4}

    # loop action
    self.loop_action_dict = {'start_button': 'game_loop', 'help_button': 'help_menu_loop', 'option_button': 'option_menu_loop', 'end_button': 'exit', 'exit': 'exit'}

    # start at main menu
    self.game_loop_state = self.game_loop_state_dict['main_menu_loop']


  def run_game(self):
    """
    run game loops
    """

    # overall loop
    while self.game_loop_state != self.game_loop_state_dict['exit']:

      # run menu loops
      if self.game_loop_state == self.game_loop_state_dict['main_menu_loop']: self.det_next_loop(self.main_menu.menu_loop())
      elif self.game_loop_state == self.game_loop_state_dict['help_menu_loop']: self.det_next_loop(self.help_menu.menu_loop())
      elif self.game_loop_state == self.game_loop_state_dict['option_menu_loop']: self.det_next_loop(self.option_menu.menu_loop())

      # run game loop
      # todo...

      # exit
      else: self.game_loop_state = self.game_loop_state_dict['exit']


  def det_next_loop(self, action):
    """
    determine next loop
    """

    # game loop state update
    self.game_loop_state = self.game_loop_state_dict[self.loop_action_dict[action]] if self.game_loop_state == self.game_loop_state_dict['main_menu_loop'] else self.game_loop_state_dict['main_menu_loop']



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