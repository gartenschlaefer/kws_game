"""
kws game
"""

import pygame
import yaml

from classifier import Classifier
from mic import Mic
from capture_screen import ScreenCapturer

# append paths
import sys
sys.path.append("./game")

# game stuff
from color_bag import ColorBag
from game_logic import ThingsGameLogic
from levels import Level_01, Level_02
from text import Text


if __name__ == '__main__':
  """
  kws game main
  """

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))


  # --
  # mic

  # windowing samples
  N, hop = int(cfg['feature_params']['N_s'] * cfg['feature_params']['fs']), int(cfg['feature_params']['hop_s'] * cfg['feature_params']['fs'])

  # create classifier
  classifier = Classifier(file=cfg['classifier']['file'], verbose=cfg['classifier']['verbose'])

  # create mic instance
  mic = Mic(fs=cfg['feature_params']['fs'], N=N, hop=hop, classifier=classifier, energy_thres=1e-4, device=8, is_audio_record=cfg['game']['capture_enabled'])


  # --
  # game setup

  # init pygame
  pygame.init()

  # init display
  screen = pygame.display.set_mode(cfg['game']['screen_size'])

  # init screen capturer
  screen_capturer = ScreenCapturer(screen, cfg['game']['screen_size'], cfg['game']['fps'], capture_path=cfg['game']['capture_path'], enabled=cfg['game']['capture_enabled'])

  # collection of game colors
  color_bag = ColorBag()
  text = Text(screen, color_bag)

  # level creation
  levels = [Level_01(screen, cfg['game']['screen_size'], color_bag, mic), Level_02(screen, cfg['game']['screen_size'], color_bag, mic)]

  # choose level
  level = levels[0]

  # game logic with dependencies
  game_logic = ThingsGameLogic(level, levels, text)

  # add clock
  clock = pygame.time.Clock()

  # mic stream and update
  with mic.stream:

    # game loop
    while game_logic.run_loop:
      for event in pygame.event.get():

        # input handling
        game_logic.event_update(event)
        level.event_update(event)

      # frame update
      level = game_logic.update()
      level.update()
      text.update()
      screen_capturer.update()

      # update display
      pygame.display.flip()

      # reduce framerate
      clock.tick(cfg['game']['fps'])

  # save video plus audio
  screen_capturer.save_video(mic)

  # end pygame
  pygame.quit()
