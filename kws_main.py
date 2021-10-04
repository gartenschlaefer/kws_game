"""
kws game main file
"""

import pygame
import yaml

from classifier import Classifier
from mic import Mic

# append paths
import sys
sys.path.append("./game")

# game stuff
from game_handler import GameHandler


if __name__ == '__main__':
  """
  kws game main
  """

  # yaml config file
  cfg = yaml.safe_load(open("./config.yaml"))

  # create classifier
  classifier = Classifier(cfg_classifier=cfg['classifier'])

  # create mic instance
  mic = Mic(classifier=classifier, mic_params=cfg['mic_params'], is_audio_record=cfg['game']['capture_enabled'])

  # game handler
  game_handler = GameHandler(cfg, mic)

  # run game
  game_handler.run_game()

  # end pygame
  pygame.quit()
