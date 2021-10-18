"""
game handler
"""

import pygame

from game_logic import GameLogic, ThingsGameLogic
from menu import MainMenu, HelpMenu, OptionMenu
from levels import Level_01, Level_02, LevelHandler, LevelCredits

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
    self.option_menu = OptionMenu(self.cfg['game'], self.screen, self.mic)

    # game loop
    self.game_loop_state_dict = {'main_menu_loop': 0, 'game_loop': 1, 'help_menu_loop': 2, 'option_menu_loop': 3, 'exit': 4}

    # loop action
    self.loop_action_dict = {'open_option_menu': 'option_menu_loop', 'open_main_menu': 'main_menu_loop', 'escape_game': 'main_menu_loop', 'start_game': 'game_loop', 'open_help_menu': 'help_menu_loop', 'exit': 'exit'}

    # start at main menu
    self.game_loop_state = self.game_loop_state_dict['main_menu_loop']


  def run_game(self):
    """
    run game loops
    """

    # overall loop
    while self.game_loop_state != self.game_loop_state_dict['exit']:

      # run menu loops
      if self.game_loop_state == self.game_loop_state_dict['main_menu_loop']: self.det_next_loop(self.main_menu.menu_loop(self.screen_capturer))
      elif self.game_loop_state == self.game_loop_state_dict['help_menu_loop']: self.det_next_loop(self.help_menu.menu_loop(self.screen_capturer))
      elif self.game_loop_state == self.game_loop_state_dict['option_menu_loop']: self.det_next_loop(self.option_menu.menu_loop(self.screen_capturer))

      # run game loop
      elif self.game_loop_state == self.game_loop_state_dict['game_loop']: self.det_next_loop(self.game_loop())

      # exit
      else: self.game_loop_state = self.game_loop_state_dict['exit']

    # save video plus audio
    self.screen_capturer.save_video(self.mic)


  def det_next_loop(self, action):
    """
    determine next loop
    """

    print("action: ", action)

    # game loop state update
    self.game_loop_state = self.game_loop_state_dict[self.loop_action_dict[action]]


  def game_loop(self):
    """
    actual game loop
    """

    # level creation
    levels = [Level_01(self.screen, self.cfg['game']['screen_size'], self.mic), Level_02(self.screen, self.cfg['game']['screen_size'], self.mic), LevelCredits(self.screen, self.cfg['game']['screen_size'])]
    #levels = [Level_01(self.screen, self.cfg['game']['screen_size'], self.mic)]

    # level handler
    level_handler = LevelHandler(levels=levels, start_level=0)

    # add clock
    clock = pygame.time.Clock()

    # init stream
    self.mic.init_stream()
    
    # mic stream and update
    with self.mic.stream:

      # game loop
      while level_handler.runs():
        for event in pygame.event.get():

          # input handling
          level_handler.event_update(event)
          self.screen_capturer.event_update(event)

        # frame update
        level_handler.update()
        self.screen_capturer.update()

        # update display
        pygame.display.flip()

        # reduce framerate
        clock.tick(self.cfg['game']['fps'])

    return 'escape_game' if not level_handler.quit() else 'exit'



if __name__ == '__main__':
  """
  main
  """

  import yaml

  # append paths
  import sys
  sys.path.append("../")

  from classifier import Classifier
  from mic import Mic

  # yaml config file
  cfg = yaml.safe_load(open("../config.yaml"))

  # create classifier
  classifier = Classifier(cfg_classifier=cfg['classifier'], root_path='../')
  
  # create mic instance
  mic = Mic(classifier=classifier, mic_params=cfg['mic_params'], is_audio_record=True)


  # game handler
  game_handler = GameHandler(cfg, mic)

  # run game
  game_handler.run_game()

  # end pygame
  pygame.quit()