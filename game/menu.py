"""
menues
"""

import pygame
import pathlib

from glob import glob

from color_bag import ColorBag
from interactable import Interactable
from game_logic import MenuGameLogic
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
    self.text.render_small_msg('menu', (0, 0))

    # game logic
    self.game_logic = MenuGameLogic(self)

    # actual up down click
    self.ud_click = 0

    # click
    self.click = False

    # selection
    self.sel = 0

    # button dict, selection: applications
    self.button_dict = {0: 'start', 1: 'options', 2: 'exit'}

    # max selection
    self.max_sel = len(self.button_dict)


  def button_state_update(self):
    """
    button state
    """

    # check if clicked
    if not self.click:

      # up
      if self.ud_click < 0 and self.sel: self.sel -= 1

      # down
      elif self.ud_click > 0 and self.sel < self.max_sel: self.sel += 1

      # set click
      self.click = True

    # reset click
    if self.ud_click == 0: self.click = False


  def button_enter(self):
    """
    button enter
    """

    print("enter button: ", self.sel)

    # start game
    if self.button_dict[self.sel] == 'start': pass

    # options
    elif self.button_dict[self.sel] == 'options': pass

    # exit
    elif self.button_dict[self.sel] == 'exit': self.game_logic.run_loop = False


  def update(self):
    """
    update menu
    """

    # fill screen
    self.screen.fill(self.color_bag.background)

    # text
    self.text.update()

    # ud click
    self.button_state_update()


  def menu_loop(self):
    """
    menu loop
    """

    # add clock
    clock = pygame.time.Clock()

    # next loop
    next_loop = 'exit'

    # game loop
    while self.game_logic.run_loop:
      for event in pygame.event.get():

        # input handling
        self.game_logic.event_update(event)

      # update menu
      self.update()

      # update display
      pygame.display.flip()

      # reduce framerate
      clock.tick(self.cfg_game['fps'])

    return next_loop



class MainMenu(Menu):
  """
  main menu
  """

  def __init__(self, cfg_game, screen):

    # Parent init
    super().__init__(cfg_game, screen)

    # button dict, selection: applications
    self.button_dict = {0: 'start', 1: 'options', 2: 'exit'}

    # max selection
    self.max_sel = len(self.button_dict)

    # text
    self.text.render_small_msg('main menu', (0, 0))

    # root for sprites
    self.button_path = str(pathlib.Path(__file__).parent.absolute()) + "/art/buttons/sprites/"

    # button images
    self.button_images = [pygame.image.load(s).convert_alpha() for s in glob(self.button_path + '*.png')]


  def button_enter(self):
    """
    button enter
    """

    print("enter button: ", self.sel)

    # start game
    if self.button_dict[self.sel] == 'start': pass

    # options
    elif self.button_dict[self.sel] == 'options': pass

    # exit
    elif self.button_dict[self.sel] == 'exit': self.game_logic.run_loop = False



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
  menu = MainMenu(cfg['game'], screen)

  # run menu loop
  menu.menu_loop()

  # end pygame
  pygame.quit()