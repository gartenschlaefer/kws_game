"""
game handler
"""

import pygame

from color_bag import ColorBag
from interactable import Interactable
from game_logic import GameLogic, ThingsGameLogic
from menu import MainMenu, HelpMenu, OptionMenu
from text import Text
from levels import Level_01, Level_02

# append paths
import sys
sys.path.append("../")
from capture_screen import ScreenCapturer


class GameHandler():
  """
  game handler
  """

  def __init__(self, cfg, mic=None):

    # arguments
    self.cfg = cfg
    self.mic = mic

    # init pygame
    pygame.init()

    # init display
    self.screen = pygame.display.set_mode(self.cfg['game']['screen_size'])

    # screen capturer
    self.screen_capturer = ScreenCapturer(self.screen, self.cfg['game'])

    # menues
    self.main_menu = MainMenu(self.cfg['game'], self.screen)
    self.help_menu = HelpMenu(self.cfg['game'], self.screen)
    self.option_menu = OptionMenu(self.cfg['game'], self.screen)

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
      elif self.game_loop_state == self.game_loop_state_dict['game_loop']: self.det_next_loop(self.game_loop())

      # exit
      else: self.game_loop_state = self.game_loop_state_dict['exit']


  def det_next_loop(self, action):
    """
    determine next loop
    """

    # game loop state update
    self.game_loop_state = self.game_loop_state_dict[self.loop_action_dict[action]] if self.game_loop_state == self.game_loop_state_dict['main_menu_loop'] else self.game_loop_state_dict['main_menu_loop']


  def game_loop(self):
    """
    actual game loop
    """

    # text
    text = Text(self.screen)

    # level creation
    levels = [Level_01(self.screen, self.cfg['game']['screen_size'], self.mic), Level_02(self.screen, self.cfg['game']['screen_size'], self.mic)]

    # choose level
    level = levels[0]

    # game logic with dependencies
    game_logic = ThingsGameLogic(level, levels, text)

    # add clock
    clock = pygame.time.Clock()

    # mic stream and update
    #with self.mic.stream:

    # game loop
    while game_logic.run_loop:
      for event in pygame.event.get():

        # input handling
        game_logic.event_update(event)
        level.event_update(event)

      # frame update
      level = game_logic.update()
      level.update()
      text.draw()
      self.screen_capturer.update()

      # update display
      pygame.display.flip()

      # reduce framerate
      clock.tick(cfg['game']['fps'])

    # save video plus audio
    self.screen_capturer.save_video(self.mic)


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