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
from input_handler import InputKeyHandler


class Menu(Interactable):
  """
  menu class
  """

  def __init__(self, cfg_game, screen, mic):

    # arguments
    self.cfg_game = cfg_game
    self.screen = screen
    self.mic = mic

    # colors
    self.color_bag = ColorBag()

    # canvas
    self.canvas = Canvas(self.screen)

    # game logic
    self.game_logic = MenuGameLogic(self)

    # interactables
    self.interactable_dict = {'game_logic': self.game_logic}

    # key handler
    self.interactable_dict.update({'input_key_handler': InputKeyHandler(objs=[self.game_logic])})

    # actual up down click
    self.ud_click = 0
    self.lr_click = 0

    # click
    self.click = False

    # define state dictionaries
    self.button_state_dict, self.state_action_dict, self.button_state = self.define_state_dicts()


  def define_state_dicts(self):
    """
    state dictionaries
    """

    # button dict, selection: button in canvas
    button_state_dict = {'start_button': 0, 'help_button': 1, 'end_button': 2}

    # action dict
    state_action_dict = {0: 'start_game', 1: 'open_help_menu', 2: 'exit'}

    return button_state_dict, state_action_dict, button_state_dict['start_button']


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
    [interactable.reset() for interactable in self.interactable_dict.values()]


  def event_update(self, event):
    """
    event update
    """
    [interactable.event_update(event) for interactable in self.interactable_dict.values()]


  def update(self):
    """
    update menu
    """

    # interactables
    [interactable.update() for interactable in self.interactable_dict.values()]

    # canvas
    self.canvas.update()
    self.canvas.draw()
    self.canvas.draw_overlay()

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


  def menu_loop(self, screen_capture=None):
    """
    menu loop
    """

    # add clock
    clock = pygame.time.Clock()

    # hack for recording: can be removed later
    #self.mic.change_device_flag = True

    while self.game_logic.run_loop:

      # init stream
      self.mic.init_stream(enable_stream=self.mic.change_device_flag)

      # mic stream and update
      with self.mic.stream:

        # run for current mic device
        while self.game_logic.run_loop:

          # events
          for event in pygame.event.get(): 
            self.event_update(event)
            if screen_capture is not None: screen_capture.event_update(event)

          # update menu
          self.update()
          if screen_capture is not None: screen_capture.update()

          # update display
          pygame.display.flip()

          # reduce framerate
          clock.tick(self.cfg_game['fps'])

          # break loop if device is changed
          if self.mic.change_device_flag: break

    # action at ending loop
    action = (self.state_action_dict[self.button_state] if not self.game_logic.esc_key_exit else self.state_action_dict[self.button_state_dict['end_button']]) if not self.game_logic.quit_game else 'exit'

    # reset game logic
    self.game_logic.reset()

    return action



class MainMenu(Menu):
  """
  main menu
  """

  def __init__(self, cfg_game, screen, mic):

    # Parent init
    super().__init__(cfg_game, screen, mic)

    # canvas
    self.canvas = CanvasMainMenu(self.screen, self.mic)

    # set button active
    self.canvas.interactable_dict['start_button'].button_press()


  def define_state_dicts(self):
    """
    state dictionaries
    """

    # button dict, selection: button in canvas
    button_state_dict = {'start_button': 0, 'help_button': 1, 'option_button': 2, 'end_button': 3}

    # action dict
    state_action_dict = {0: 'start_game', 1: 'open_help_menu', 2: 'open_option_menu', 3: 'exit'}

    return button_state_dict, state_action_dict, button_state_dict['start_button']



class HelpMenu(Menu):
  """
  main menu
  """

  def __init__(self, cfg_game, screen, mic):

    # Parent init
    super().__init__(cfg_game, screen, mic)

    # canvas
    self.canvas = CanvasHelpMenu(self.screen)

    # set button active
    self.canvas.interactable_dict['end_button'].button_press()


  def define_state_dicts(self):
    """
    state dictionaries
    """

    # button dict, selection: button in canvas
    button_state_dict = {'end_button': 0}

    # action dict
    state_action_dict = {0: 'open_main_menu', 1: 'exit'}

    return button_state_dict, state_action_dict, button_state_dict['end_button']



class OptionMenu(Menu):
  """
  main menu
  """

  def __init__(self, cfg_game, screen, mic):

    # Parent init
    super().__init__(cfg_game, screen, mic)

    # canvas
    self.canvas = CanvasOptionMenu(self.screen, self.mic)

    # set button active
    self.canvas.interactable_dict['end_button'].button_press()

    # menu buttons selection enable
    self.menu_button_sel_enable = True


  def define_state_dicts(self):
    """
    state dictionaries
    """

    # button dict, selection: button in canvas
    button_state_dict = {'cmd_button': 0, 'thresh_button': 1, 'device_button': 2, 'end_button': 3}

    # action dict
    state_action_dict = {3: 'open_main_menu', 4: 'exit'}

    return button_state_dict, state_action_dict, button_state_dict['end_button']


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
    self.canvas.draw_overlay()

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
    user_settings = yaml.safe_load(open(self.mic.user_settings_file)) if os.path.isfile(self.mic.user_settings_file) else {}

    # update energy thres
    user_settings.update({'energy_thresh_db': e})

    # write file
    with open(self.cfg_game['user_settings_file'], 'w') as f:
      yaml.dump(user_settings, f)


  def save_user_settings_select_device(self, device):
    """
    save user settings selected device
    """

    print("save user-settings")

    # load user settings
    user_settings = yaml.safe_load(open(self.mic.user_settings_file)) if os.path.isfile(self.mic.user_settings_file) else {}

    # update energy thres
    user_settings.update({'select_device': True, 'device': device})

    # write file
    with open(self.cfg_game['user_settings_file'], 'w') as f:
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