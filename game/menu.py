"""
menues
"""

import pygame
import pathlib

from glob import glob

from color_bag import ColorBag
from interactable import Interactable
from game_logic import MenuGameLogic
from canvas import Canvas, CanvasMainMenu


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

    # canvas
    self.canvas = Canvas(self.screen)

    # game logic
    self.game_logic = MenuGameLogic(self)

    # actual up down click
    self.ud_click = 0

    # click
    self.click = False

    # selection
    self.button_select = 0

    # button dict, selection: button in canvas
    self.button_dict = {0: 'start_button', 1: 'help_button', 2: 'end_button'}


  def button_state_update(self):
    """
    button state
    """

    # check if clicked
    if not self.click and self.ud_click:

      # up
      if self.ud_click < 0 and self.button_select: 
        self.button_click()
        self.button_select -= 1

      # down
      elif self.ud_click > 0 and self.button_select < len(self.button_dict) - 1: 
        self.button_click()
        self.button_select += 1

      # return
      else: return

      # set click
      self.click = True

      # set button click
      self.button_click()

    # reset click
    if self.ud_click == 0: self.click = False


  def button_click(self):
    """
    button click
    """
    pass


  def button_enter(self):
    """
    button enter
    """

    # start game
    if self.button_dict[self.button_select] == 'start_button': pass

    # options
    elif self.button_dict[self.button_select] == 'help_button': pass

    # exit
    elif self.button_dict[self.button_select] == 'end_button': self.game_logic.run_loop = False


  def update(self):
    """
    update menu
    """

    # canvas
    self.canvas.update()

    # up down movement
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

    # button dict, selection: button in canvas
    self.button_dict = {0: 'start_button', 1: 'help_button', 2: 'end_button'}

    # canvas
    self.canvas = CanvasMainMenu(self.screen)

    # set button active
    self.canvas.interactables_dict['start_button'].button_press()


  def button_click(self):
    """
    button click
    """
    
    # change button image
    self.canvas.interactables_dict[self.button_dict[self.button_select]].button_press()


  def button_enter(self):
    """
    button enter
    """

    # start game
    if self.button_dict[self.button_select] == 'start_button': pass

    # options
    elif self.button_dict[self.button_select] == 'help_button': pass

    # exit
    elif self.button_dict[self.button_select] == 'end_button': self.game_logic.run_loop = False



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