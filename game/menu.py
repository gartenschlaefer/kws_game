"""
menues
"""

import pygame
import pathlib

from glob import glob

from color_bag import ColorBag
from interactable import Interactable
from game_logic import MenuGameLogic
from canvas import Canvas, CanvasMainMenu, CanvasHelpMenu, CanvasOptionMenu


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

      # deselect buttons
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

      # select buttons
      self.button_click()

    # reset click
    if self.ud_click == 0: self.click = False


  def button_click(self):
    """
    button click
    """

    # change button image
    try:
      self.canvas.interactables_dict[self.button_dict[self.button_select]].button_press()
    except:
      print("button not available in canvas: ", self.button_dict[self.button_select])


  def button_enter(self):
    """
    button enter
    """

    # end game loop (remembers last button state)
    self.game_logic.run_loop = False


  def reset(self):
    """
    reset menu
    """

    # reset run loop
    self.game_logic.reset()


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

    # action at ending loop
    action = self.button_dict[self.button_select] if not self.game_logic.esc_key_exit else 'exit'

    # reset game logic
    self.game_logic.reset()

    return action



class MainMenu(Menu):
  """
  main menu
  """

  def __init__(self, cfg_game, screen):

    # Parent init
    super().__init__(cfg_game, screen)

    # button dict, selection: button in canvas
    self.button_dict = {0: 'start_button', 1: 'help_button', 2: 'option_button', 3: 'end_button'}

    # canvas
    self.canvas = CanvasMainMenu(self.screen)

    # set button active
    self.canvas.interactables_dict['start_button'].button_press()



class HelpMenu(Menu):
  """
  main menu
  """

  def __init__(self, cfg_game, screen):

    # Parent init
    super().__init__(cfg_game, screen)

    # button dict, selection: button in canvas
    self.button_dict = {0: 'end_button'}

    # canvas
    self.canvas = CanvasHelpMenu(self.screen)

    # set button active
    self.canvas.interactables_dict['end_button'].button_press()



class OptionMenu(Menu):
  """
  main menu
  """

  def __init__(self, cfg_game, screen):

    # Parent init
    super().__init__(cfg_game, screen)

    # button dict, selection: button in canvas
    self.button_dict = {0: 'end_button'}

    # canvas
    self.canvas = CanvasOptionMenu(self.screen)

    # set button active
    self.canvas.interactables_dict['end_button'].button_press()



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
  #menu = Menu(cfg['game'], screen)
  menu = MainMenu(cfg['game'], screen)
  #menu = HelpMenu(cfg['game'], screen)
  #menu = OptionMenu(cfg['game'], screen)

  # run menu loop
  menu.menu_loop()

  # end pygame
  pygame.quit()