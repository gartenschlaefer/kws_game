"""
menues
"""

import pygame
import pathlib

from glob import glob

from color_bag import ColorBag
from interactable import Interactable
from game_logic import MenuGameLogic
from canvas import Canvas, CanvasMainMenu, CanvasHelpMenu, CanvasOptionMenu, CanvasDevice


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
    self.lr_click = 0

    # click
    self.click = False

    # selection
    self.button_state = 0

    # button dict, selection: button in canvas
    self.button_dict = {0: 'start_button', 1: 'help_button', 2: 'end_button'}


  def direction_change(self, direction):
    """
    arrow keys pressed
    """
    
    # ud click state
    self.ud_click += direction[1]
    self.lr_click += direction[0]


  def enter_key(self):
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


  def event_update(self, event):
    """
    event update
    """

    # game logic
    self.game_logic.event_update(event)


  def update(self):
    """
    update menu
    """

    # canvas
    self.canvas.update()
    self.canvas.draw()

    # up down movement
    self.button_state_update()


  def button_state_update(self):
    """
    button state
    """

    # check if clicked
    if not self.click and self.ud_click:

      # deselect buttons
      if self.ud_click < 0 and self.button_state: 
        self.button_deselect()
        self.button_state -= 1

      # down
      elif self.ud_click > 0 and self.button_state < len(self.button_dict) - 1: 
        self.button_deselect()
        self.button_state += 1

      # return
      else: return

      # set click
      self.click = True

      # select buttons
      self.button_select()

    # reset click
    if self.ud_click == 0: self.click = False


  def button_select(self):
    """
    button select
    """
    self.button_toggle()


  def button_deselect(self):
    """
    button select
    """
    self.button_toggle()


  def button_toggle(self):
    """
    button click
    """

    # change button image
    try:
      self.canvas.interactable_dict[self.button_dict[self.button_state]].button_press()
    except:
      print("button not available in canvas: ", self.button_dict[self.button_state])


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
        self.event_update(event)

      # update menu
      self.update()

      # update display
      pygame.display.flip()

      # reduce framerate
      clock.tick(self.cfg_game['fps'])

    # action at ending loop
    action = self.button_dict[self.button_state] if not self.game_logic.esc_key_exit else 'exit'

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
    self.canvas.interactable_dict['start_button'].button_press()



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
    self.canvas.interactable_dict['end_button'].button_press()



class OptionMenu(Menu):
  """
  main menu
  """

  def __init__(self, cfg_game, screen, mic):

    # Parent init
    super().__init__(cfg_game, screen)

    # arguments
    self.mic = mic

    # button dict, selection: button in canvas
    self.button_dict = {0: 'cmd_button', 1: 'thresh_button', 2: 'device_button', 3: 'end_button'}

    # selection
    self.button_state = 3

    # canvas
    self.canvas = CanvasOptionMenu(self.screen, self.mic)

    # set button active
    self.canvas.interactable_dict['end_button'].button_press()

    # device canvas
    self.canvas.interactable_dict.update({'device_canvas': CanvasDevice(self.canvas.canvas_surf, size=(200, 200), position=(200, 200))})

    # put device canvas first to render
    self.canvas.interactable_dict = {k:v for k, v in list(self.canvas.interactable_dict.items())[-1:] + list(self.canvas.interactable_dict.items())[:-1]}


  def menu_loop(self):
    """
    menu loop
    """

    # add clock
    clock = pygame.time.Clock()

    # mic stream and update
    with self.mic.stream:
      while self.game_logic.run_loop:
        for event in pygame.event.get():

          # input handling
          self.event_update(event)

        # update menu
        self.update()

        # update display
        pygame.display.flip()

        # reduce framerate
        clock.tick(self.cfg_game['fps'])

    # action at ending loop
    action = self.button_dict[self.button_state] if not self.game_logic.esc_key_exit else 'exit'

    # reset game logic
    self.game_logic.reset()

    return action


  def enter_key(self):
    """
    button enter
    """

    # end loop
    if self.button_dict[self.button_state] == 'end_button': self.game_logic.run_loop = False


  def button_select(self):
    """
    button select
    """

    # toggle button image
    self.button_toggle()

    # activate dev page
    if self.button_dict[self.button_state] == 'device_button':
      self.toggle_device_canvas()
      print("open device page")


  def button_deselect(self):
    """
    button deselect
    """

    # toggle button image
    self.button_toggle()

    if self.button_dict[self.button_state] == 'device_button': 
      self.toggle_device_canvas()
      print("close device page")


  def toggle_device_canvas(self):
    """
    device page
    """

    self.canvas.interactable_dict['device_canvas'].enabled = not self.canvas.interactable_dict['device_canvas'].enabled


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
  mic = Mic(classifier=classifier, feature_params=cfg['feature_params'], mic_params=cfg['mic_params'], is_audio_record=True)

  # init pygame
  pygame.init()

  # init display
  screen = pygame.display.set_mode(cfg['game']['screen_size'])

  # menu
  #menu = Menu(cfg['game'], screen)
  #menu = MainMenu(cfg['game'], screen)
  #menu = HelpMenu(cfg['game'], screen)
  menu = OptionMenu(cfg['game'], screen, mic)

  # run menu loop
  menu.menu_loop()

  # end pygame
  pygame.quit()