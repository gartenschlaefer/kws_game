"""
menues for the game
"""

import pygame
import pathlib
import os
import yaml

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
    self.lr_click = 0

    # click
    self.click = False

    # selection
    self.button_state = 0

    # button dict, selection: button in canvas
    self.button_state_dict = {'start_button': 0, 'help_button': 1, 'end_button': 2}

    # set button state
    self.button_state = self.button_state_dict['start_button']


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


  def esc_key(self):
    """
    esc key
    """

    # end loop
    self.game_logic.run_loop = False

    # end with escape key pressed
    self.game_logic.esc_key_exit = True


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
      elif self.ud_click > 0 and self.button_state < len(self.button_state_dict) - 1: 
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
      self.canvas.interactable_dict[list(self.button_state_dict.keys())[list(self.button_state_dict.values()).index(self.button_state)]].button_press()
    except:
      print("button not available in canvas: ", list(self.button_state_dict.keys())[list(self.button_state_dict.values()).index(self.button_state)])


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
    action = list(self.button_state_dict.keys())[list(self.button_state_dict.values()).index(self.button_state)] if not self.game_logic.esc_key_exit else 'exit'

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
    self.button_state_dict = {'start_button': 0, 'help_button': 1, 'option_button': 2, 'end_button': 3}

    # set button state
    self.button_state = self.button_state_dict['start_button']

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
    self.button_state_dict = {'end_button': 0}

    # set button state
    self.button_state = self.button_state_dict['end_button']

    # canvas
    self.canvas = CanvasHelpMenu(self.screen)

    # set button active
    self.canvas.interactable_dict['end_button'].button_press()



class OptionMenu(Menu):
  """
  main menu
  """

  def __init__(self, cfg_game, screen, mic, root_path='./'):

    # Parent init
    super().__init__(cfg_game, screen)

    # arguments
    self.mic = mic
    self.root_path = root_path

    # button dict, selection: button in canvas
    self.button_state_dict = {'cmd_button': 0, 'thresh_button': 1, 'device_button': 2, 'end_button': 3}

    # selection
    self.button_state = self.button_state_dict['end_button']

    # canvas
    self.canvas = CanvasOptionMenu(self.screen, self.mic)

    # set button active
    self.canvas.interactable_dict['end_button'].button_press()

    # menu buttons selection enable
    self.menu_button_sel_enable = True

    # user settings file
    self.user_setting_file = self.root_path + self.cfg_game['user_setting_file']


  def menu_loop(self):
    """
    menu loop
    """

    # add clock
    clock = pygame.time.Clock()

    while self.game_logic.run_loop:

      print("new mic loop")

      # user settings
      self.mic.load_user_settings(self.user_setting_file)

      # init stream
      self.mic.init_stream(enable_stream=self.mic.change_device_flag)

      # mic stream and update
      with self.mic.stream:

        # run for current mic device
        while self.game_logic.run_loop:

          # events
          for event in pygame.event.get(): self.event_update(event)

          # update menu
          self.update()

          # update display
          pygame.display.flip()

          # reduce framerate
          clock.tick(self.cfg_game['fps'])

          # break loop if device is changed
          if self.mic.change_device_flag: break

    # action at ending loop
    action = list(self.button_state_dict.keys())[list(self.button_state_dict.values()).index(self.button_state)] if not self.game_logic.esc_key_exit else 'exit'

    # reset game logic
    self.game_logic.reset()

    return action


  def enter_key(self):
    """
    button enter
    """

    # update selection mode
    self.menu_button_sel_enable = not self.menu_button_sel_enable
    
    # end loop
    if self.button_state == self.button_state_dict['end_button']: 

      # end loop
      self.game_logic.run_loop = False

      # button sel state
      self.menu_button_sel_enable = True

    # device menu
    elif self.button_state == self.button_state_dict['device_button']:

      # set device canvas active for selection
      self.canvas.interactable_dict['device_canvas'].device_select(not self.menu_button_sel_enable)

      # enter in device select
      if self.menu_button_sel_enable:

        # update mic device
        self.mic.change_device(self.canvas.interactable_dict['device_canvas'].active_device_num)

        # save device
        self.save_user_settings_select_device(self.canvas.interactable_dict['device_canvas'].active_device_num)

    # thresh menu
    elif self.button_state == self.button_state_dict['thresh_button']: 

      # select energy to change
      self.canvas.interactable_dict['thresh_canvas'].select(not self.menu_button_sel_enable)

      # enter in device select
      if self.menu_button_sel_enable:

        # update mic device
        self.mic.change_energy_thresh_db(self.canvas.interactable_dict['thresh_canvas'].energy_thresh_db)

        # save device
        self.save_user_settings_thresh(self.canvas.interactable_dict['thresh_canvas'].energy_thresh_db)

      # activate mic device
      else: self.mic.change_device_flag = True

    # cmd menu
    elif self.button_state == self.button_state_dict['cmd_button']: 

      # select energy to change
      self.canvas.interactable_dict['cmd_canvas'].select(not self.menu_button_sel_enable)

      # activate mic device
      if not self.menu_button_sel_enable: self.mic.change_device_flag = True


  def esc_key(self):
    """
    esc key
    """

    # only if enter key was pressed
    if not self.menu_button_sel_enable:

      # update selection mode
      self.menu_button_sel_enable = not self.menu_button_sel_enable

      # device menu
      if self.button_state == self.button_state_dict['device_button']: self.canvas.interactable_dict['device_canvas'].device_select(not self.menu_button_sel_enable)

      # thresh menu
      elif self.button_state == self.button_state_dict['thresh_button']: self.canvas.interactable_dict['thresh_canvas'].select(not self.menu_button_sel_enable)

      # cmd menu
      elif self.button_state == self.button_state_dict['cmd_button']: self.canvas.interactable_dict['cmd_canvas'].select(not self.menu_button_sel_enable)

    # standard esc routine
    else:
      
      # end loop
      self.game_logic.run_loop = False

      # end with escape key pressed
      self.game_logic.esc_key_exit = True


  def button_select(self):
    """
    button select
    """

    # toggle button image
    self.button_toggle()

    # device button
    if self.button_state == self.button_state_dict['device_button']:

      # toggle canvas
      self.canvas.interactable_dict['device_canvas'].enabled = not self.canvas.interactable_dict['device_canvas'].enabled

      # update devices
      if self.canvas.interactable_dict['device_canvas'].enabled: self.canvas.interactable_dict['device_canvas'].devices_to_text(), self.canvas.interactable_dict['mic_bar'].reload_thresh()

    # thresh button
    elif self.button_state == self.button_state_dict['thresh_button']:

      # toggle canvas
      self.canvas.interactable_dict['thresh_canvas'].enabled = not self.canvas.interactable_dict['thresh_canvas'].enabled

      # update energy thresh
      if self.canvas.interactable_dict['thresh_canvas'].enabled: self.canvas.interactable_dict['thresh_canvas'].reload_thresh(), self.canvas.interactable_dict['mic_bar'].reload_thresh()

    # cmd button
    elif self.button_state == self.button_state_dict['cmd_button']:

      # toggle canvas
      self.canvas.interactable_dict['cmd_canvas'].enabled = not self.canvas.interactable_dict['cmd_canvas'].enabled


  def button_deselect(self):
    """
    button deselect, here same as button select
    """
    self.button_select()


  def update(self):
    """
    update menu
    """

    # canvas
    self.canvas.update()
    self.canvas.draw()

    # up down movement
    if self.menu_button_sel_enable: self.button_state_update()

    # options
    else: 
      if self.button_state == self.button_state_dict['device_button']: self.device_menu_update()
      elif self.button_state == self.button_state_dict['thresh_button']: self.thresh_menu_update()


  def thresh_menu_update(self):
    """
    thresh menu
    """

    # check if clicked
    if not self.click and self.ud_click:

      # change thresh
      self.canvas.interactable_dict['thresh_canvas'].change_energy_thresh_key(self.ud_click)
      self.canvas.interactable_dict['mic_bar'].change_energy_thresh_db_pos(self.canvas.interactable_dict['thresh_canvas'].energy_thresh_db)

      # set click
      self.click = True

    # reset click
    if self.ud_click == 0: self.click = False


  def device_menu_update(self):
    """
    device menu
    """

    # check if clicked
    if not self.click and self.ud_click:

      # up
      if self.ud_click < 0 and self.canvas.interactable_dict['device_canvas'].active_device_id: 
        self.canvas.interactable_dict['device_canvas'].device_select(False)
        self.canvas.interactable_dict['device_canvas'].active_device_id -= 1

      # down
      elif self.ud_click > 0 and self.canvas.interactable_dict['device_canvas'].active_device_id < len(self.canvas.interactable_dict['device_canvas'].device_id_dict) - 1: 
        self.canvas.interactable_dict['device_canvas'].device_select(False)
        self.canvas.interactable_dict['device_canvas'].active_device_id += 1

      # return
      else: return

      # set click
      self.click = True

      # select device
      self.canvas.interactable_dict['device_canvas'].device_select(True)

    # reset click
    if self.ud_click == 0: self.click = False


  def save_user_settings_thresh(self, e):
    """
    save user settings energy thresh value
    """

    print("save user-settings")

    # load user settings
    user_settings = yaml.safe_load(open(self.user_setting_file)) if os.path.isfile(self.user_setting_file) else {}

    # update energy thres
    user_settings.update({'energy_thresh_db': e})

    # write file
    with open(self.cfg_game['user_setting_file'], 'w') as f:
      yaml.dump(user_settings, f)


  def save_user_settings_select_device(self, device):
    """
    save user settings selected device
    """

    print("save user-settings")

    # load user settings
    user_settings = yaml.safe_load(open(self.user_setting_file)) if os.path.isfile(self.user_setting_file) else {}

    # update energy thres
    user_settings.update({'select_device': True, 'device': device})

    # write file
    with open(self.cfg_game['user_setting_file'], 'w') as f:
      yaml.dump(user_settings, f)



if __name__ == '__main__':
  """
  main
  """

  # append paths
  import sys
  sys.path.append("../")

  from classifier import Classifier
  from mic import Mic

  # yaml config file
  cfg = yaml.safe_load(open('../config.yaml'))

  # create classifier
  classifier = Classifier(cfg_classifier=cfg['classifier'], root_path='../')
  
  # create mic instance
  mic = Mic(classifier=classifier, mic_params=cfg['mic_params'], is_audio_record=True)

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