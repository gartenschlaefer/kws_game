"""
kws game
"""

import pygame

from classifier import Classifier
from mic import Mic

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

  # --
  # mic

  # params
  fs = 16000

  # window and hop size
  N, hop = int(0.025 * fs), int(0.010 * fs)

  # create classifier
  classifier = Classifier(file='./models/fstride_c-5.npz', verbose=False)

  # create mic instance
  mic = Mic(fs=fs, N=N, hop=hop, classifier=classifier, energy_thres=1e-4, device=7)


  # --
  # game setup

  # size of display
  screen_size = width, height = 640, 480

  # init pygame
  pygame.init()

  # init display
  screen = pygame.display.set_mode(screen_size)

  # collection of game colors
  color_bag = ColorBag()
  text = Text(screen, color_bag)

  # level creation
  levels = [Level_01(screen, screen_size, color_bag, mic), Level_02(screen, screen_size, color_bag, mic)]

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

      # update display
      pygame.display.flip()

      # reduce framerate
      clock.tick(60)

    # end pygame
    pygame.quit()
