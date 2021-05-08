"""
menues
"""

import pygame

from color_bag import ColorBag
from interactable import Interactable
from game_logic import GameLogic
from text import Text

class Menu(Interactable):
  """
  menu class
  """

  def __init__(self, cfg_game, screen):

    # arguments
    self.cfg_game = cfg_game
    self.screen = screen

    # colors
    self.color_bag = ColorBag()

    # text
    self.text = Text(self.screen)
    self.text.render_small_msg('main menu', (0, 0))


  def update(self):
    """
    update menu
    """

    # fill screen
    self.screen.fill(self.color_bag.background)

    # text
    self.text.update()


  def menu_loop(self):
    """
    menu loop
    """

    # add clock
    clock = pygame.time.Clock()

    # game logic
    game_logic = GameLogic()

    # game loop
    while game_logic.run_loop:
      for event in pygame.event.get():

        # input handling
        game_logic.event_update(event)

      # update menu
      self.update()

      # update display
      pygame.display.flip()

      # reduce framerate
      clock.tick(self.cfg_game['fps'])


if __name__ == '__main__':
  """
  main
  """

  import yaml

  # yaml config file
  cfg = yaml.safe_load(open("../config.yaml"))


  # init pygame
  pygame.init()

  # init display
  screen = pygame.display.set_mode(cfg['game']['screen_size'])

  # menu
  menu = Menu(cfg['game'], screen)

  # run menu loop
  menu.menu_loop()

  # end pygame
  pygame.quit()