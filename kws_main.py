"""
kws game
"""

import pygame
import numpy as np

from classifier import Classifier
from mic import Mic

# append paths
import sys
sys.path.append("./game")

from character import Character 
from color_bag import ColorBag
from grid_world import GridWorld
from levels import setup_level_square


if __name__ == '__main__':
  """
  Main Gridworld
  """

  # size of display
  screen_size = width, height = 640, 480

  # some vars
  run_loop = True

  # collection of game colors
  color_bag = ColorBag()

  # init pygame
  pygame.init()

  # init display
  screen = pygame.display.set_mode(screen_size)

  # sprite groups
  all_sprites = pygame.sprite.Group()


  # --
  # mic

  # params
  fs = 16000

  # window and hop size
  N, hop = int(0.025 * fs), int(0.010 * fs)

  # create classifier
  classifier = Classifier(file='./ignore/models/best_models/fstride_c-5.npz')  

  # create mic instance
  mic = Mic(fs=fs, N=N, hop=hop, classifier=classifier)


  # create gridworld
  grid_world = GridWorld(screen_size, color_bag, mic)
  setup_level_square(grid_world)

  # add sprites
  all_sprites.add(grid_world.wall_sprites, grid_world.move_wall_sprites)

  # add clock
  clock = pygame.time.Clock()

  # mic stream and update
  with mic.stream:

    # game loop
    while run_loop:
      for event in pygame.event.get():

        # input handling in grid world
        run_loop = grid_world.event_update(event, run_loop)

      # frame update
      grid_world.frame_update()

      # update sprites
      all_sprites.update()

      # fill screen
      screen.fill(color_bag.background)

      # draw sprites
      all_sprites.draw(screen)

      # update display
      pygame.display.flip()

      # reduce framerate
      clock.tick(60)

  # end pygame
  pygame.quit()
