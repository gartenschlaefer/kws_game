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

  # create classifier
  classifier = Classifier(model_path=cfg['classifier']['model_path'], verbose=cfg['classifier']['verbose'])

  # create mic instance
  mic = Mic(classifier=classifier, feature_params=cfg['feature_params'], mic_params=cfg['mic_params'], is_audio_record=cfg['game']['capture_enabled'])

  
  # --
  # game setup

  # init pygame
  pygame.init()

  # init display
  screen = pygame.display.set_mode(cfg['game']['screen_size'])

  # init screen capturer
  screen_capturer = ScreenCapturer(screen, cfg['game']['screen_size'], cfg['game']['fps'], capture_path=cfg['game']['capture_path'], enabled=cfg['game']['capture_enabled'])

  # text
  text = Text(screen)

  # level creation
  levels = [Level_01(screen, cfg['game']['screen_size'], mic), Level_02(screen, cfg['game']['screen_size'], mic)]

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
