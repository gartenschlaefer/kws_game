"""
canvas, a space for creativity
"""

import numpy as np
import pygame

from interactable import Interactable
from text import Text
from button import StartButton, EndButton, HelpButton, OptionButton, DeviceButton, ThreshButton, CmdButton
from color_bag import ColorBag
from mic_bar import MicBar
from things import Spaceship


class Canvas(Interactable):
  """
  Canvas base class
  """

  def __init__(self, screen, size=None, position=(0, 0)):

    # arguments
    self.screen = screen
    self.size = size if size is not None else self.screen.get_size()
    self.position = position

    # canvas surface (use same size as screen)
    self.canvas_surf = pygame.Surface(self.size)

    # colorbag
    self.color_bag = ColorBag()

    # color
    self.color_background = self.color_bag.canvas_background

    # interactables
    self.interactable_dict = {}

    # enabled
    self.enabled = True


  def reset(self):
    """
    reset level
    """

    # reenable
    self.enabled = True

    # interactables reset
    for interactable in self.interactable_dict.values(): interactable.reset()


  def event_update(self, event):
    """
    event update
    """

    if not self.enabled: return

    # interactables
    for interactable in self.interactable_dict.values(): interactable.event_update(event)


  def update(self):
    """
    update
    """
    
    if not self.enabled: return

    # update all interactables
    for interactable in self.interactable_dict.values(): interactable.update()


  def draw(self):
    """
    draw
    """

    if not self.enabled: return

    # fill screen
    self.canvas_surf.fill(self.color_background)

    # blit stuff
    for interactable in self.interactable_dict.values(): interactable.draw()

    # draw canvas
    self.screen.blit(self.canvas_surf, self.position)



class CanvasMainMenu(Canvas):
  """
  main menu canvas
  """

  def __init__(self, screen):

    # Parent init
    super().__init__(screen)

    # add text
    text = Text(self.canvas_surf, message='main menu', position=(0, 0), font_size='small', color=self.color_bag.text_menu)

    # update canvas objects
    self.interactable_dict.update({'text': text, 'start_button': StartButton(self.canvas_surf, position=(30, 75), scale=(3, 3)), 'help_button': HelpButton(self.canvas_surf, position=(30, 175), scale=(3, 3)), 'option_button': OptionButton(self.canvas_surf, position=(30, 275), scale=(3, 3)), 'end_button': EndButton(self.canvas_surf, position=(30, 375), scale=(3, 3))})



class CanvasHelpMenu(Canvas):
  """
  main menu canvas
  """

  def __init__(self, screen):

    # Parent init
    super().__init__(screen)

    # add text
    self.interactable_dict.update({'text_menu_info': Text(self.canvas_surf, message='help', position=(0, 0), font_size='small', color=self.color_bag.text_menu)})
    self.interactable_dict.update({'text_help1': Text(self.canvas_surf, message='Movement: Arrow keys and Space for jump', position=(40, 50 + 30), font_size='tiny_small', color=self.color_bag.text_menu)})
    self.interactable_dict.update({'text_help2': Text(self.canvas_surf, message='A Microphone is needed to capture speech commands', position=(40, 50 + 60), font_size='tiny_small', color=self.color_bag.text_menu)})
    self.interactable_dict.update({'text_help3': Text(self.canvas_surf, message='Checkout the option menu for your Mircrophone settings', position=(40, 50 + 90), font_size='tiny_small', color=self.color_bag.text_menu)})
    self.interactable_dict.update({'text_help4': Text(self.canvas_surf, message='Use speech commands [left, right, up, down, go]', position=(40, 50 + 120), font_size='tiny_small', color=self.color_bag.text_menu)})
    self.interactable_dict.update({'text_credits': Text(self.canvas_surf, message='Credits: Christian Walter', position=(40, 50 + 200), font_size='tiny_small', color=self.color_bag.text_menu)})

    # end button
    self.interactable_dict.update({'end_button': EndButton(self.canvas_surf, position=(30, 375), scale=(3, 3))})



class CanvasOptionMenu(Canvas):
  """
  option menu canvas
  """

  def __init__(self, screen, mic):

    # Parent init
    super().__init__(screen)

    # arguments
    self.mic = mic

    # add text
    text = Text(self.canvas_surf, message='options', position=(0, 0), font_size='small', color=self.color_bag.text_menu)

    # mic bar
    mic_bar = MicBar(self.canvas_surf, self.mic, position=(540, 225), bar_size=(30, 150), scale_margin=(50, 40))

    # device canvas
    self.interactable_dict.update({'device_canvas': CanvasDevice(self.canvas_surf, self.mic, size=(350, 380), position=(170, 50))})

    # thresh canvas
    self.interactable_dict.update({'thresh_canvas': CanvasThresh(self.canvas_surf, self.mic, size=(350, 380), position=(170, 50))})
    
    # command canvas
    self.interactable_dict.update({'cmd_canvas': CanvasCommand(self.canvas_surf, self.mic, size=(350, 380), position=(170, 50))})

    # update canvas objects
    self.interactable_dict.update({'text': text, 'mic_bar': mic_bar, 'cmd_button': CmdButton(self.canvas_surf, position=(30, 75), scale=(3, 3)), 'thresh_button': ThreshButton(self.canvas_surf, position=(30, 175), scale=(3, 3)), 'device_button': DeviceButton(self.canvas_surf, position=(30, 275), scale=(3, 3)), 'end_button': EndButton(self.canvas_surf, position=(30, 375), scale=(3, 3))})



class CanvasCommand(Canvas):
  """
  speech commands canvas
  """

  def __init__(self, screen, mic, size, position):

    # Parent init
    super().__init__(screen, size=size, position=position)

    # arguments
    self.mic = mic

    # set background color
    self.color_background = self.color_bag.canvas_option_background

    # enable view
    self.enabled = False

    # num kws cmds
    self.num_kws_cmds = 5

    # info text
    self.interactable_dict.update({'text_info': Text(self.canvas_surf, message='speech commands: ', position=(0, 0), font_size='small', color=self.color_bag.text_menu)})
    
    # class dict text
    self.interactable_dict.update({'text_model': Text(self.canvas_surf, message='model: {}'.format(self.mic.classifier.nn_arch), position=(20, 40), font_size='tiny', color=self.color_bag.text_menu)})
    self.interactable_dict.update({'text_class_dict': Text(self.canvas_surf, message='key words:', position=(20, 60), font_size='tiny', color=self.color_bag.text_menu)})

    # class text
    self.interactable_dict.update({'text_class{}'.format(v): Text(self.canvas_surf, message='{}'.format(k), position=(30 + 90 * int(v > 5), 80 + 15 * v - 90 * int(v > 5)), font_size='tiny', color=self.color_bag.text_menu) for (k, v) in self.mic.classifier.class_dict.items()})

    # kws text
    self.interactable_dict.update({'text_kws': Text(self.canvas_surf, message='key word spotting:', position=(20, 210), font_size='tiny_small', color=self.color_bag.text_menu_active, enabled=False)})    
    self.interactable_dict.update({'text_cmd{}'.format(i): Text(self.canvas_surf, message='_', position=(50, 240 + 18 * i), font_size='tiny_small', color=self.color_bag.text_menu_active, enabled=False) for i in range(0, self.num_kws_cmds)})
    #self.interactable_dict.update({'text_cmd0': Text(self.canvas_surf, message='_', position=(50, 240), font_size='tiny_small', color=self.color_bag.text_menu_active, enabled=False)})


  def select(self, active):
    """
    select by clicking enter -> change color and make it editable
    """

    # reset texts
    if active:
      for i in range(self.num_kws_cmds):
        self.interactable_dict['text_cmd{}'.format(i)].message = '_'
        self.interactable_dict['text_cmd{}'.format(i)].render()

    # enable text
    self.interactable_dict['text_kws'].enabled = True if active else False

    # enable cmds
    for i in range(self.num_kws_cmds): self.interactable_dict['text_cmd{}'.format(i)].enabled = True if active else False


  def update(self):
    """
    update
    """
    
    # return if not enabled
    if not self.enabled: return

    # update all interactables
    for interactable in self.interactable_dict.values(): interactable.update()

    # get command
    command = self.mic.update_read_command()

    # interpret command
    if command is not None:
      print("command: ", command)

      # listing downwards
      for i in range(self.num_kws_cmds - 1, 0, -1):
        
        # update render message
        self.interactable_dict['text_cmd{}'.format(i)].message = self.interactable_dict['text_cmd{}'.format(i-1)].message
        self.interactable_dict['text_cmd{}'.format(i)].render()

      # first entry is always the command
      self.interactable_dict['text_cmd0'].message = command
      self.interactable_dict['text_cmd0'].render()



class CanvasThresh(Canvas):
  """
  energy threshold canvas
  """

  def __init__(self, screen, mic, size, position):

    # Parent init
    super().__init__(screen, size=size, position=position)

    # arguments
    self.mic = mic

    # set background color
    self.color_background = self.color_bag.canvas_option_background

    # enable view
    self.enabled = False

    # energy thresh
    self.energy_thresh_db = self.mic.mic_params['energy_thresh_db']

    # info text
    self.interactable_dict.update({'text_info': Text(self.canvas_surf, message='energy threshold: ', position=(0, 0), font_size='small', color=self.color_bag.text_menu)})
    
    # energy text
    self.interactable_dict.update({'text_energy': Text(self.canvas_surf, message='e: {:.1f}dB'.format(self.energy_thresh_db), position=(20, 40), font_size='tiny', color=self.color_bag.text_menu)})


  def reload_thresh(self):
    """
    reload thresh from mic
    """

    # load thresh from mic
    self.energy_thresh_db = self.mic.mic_params['energy_thresh_db']

    # update render message
    self.interactable_dict['text_energy'].message = 'e: {:.1f}dB'.format(self.energy_thresh_db)
    self.interactable_dict['text_energy'].render()


  def select(self, active):
    """
    select by clicking enter -> change color and make it editable
    """

    # set color and render
    self.interactable_dict['text_energy'].color = self.color_bag.text_menu_active if active else self.color_bag.text_menu
    self.interactable_dict['text_energy'].render()


  def change_energy_thresh_key(self, ud=0):
    """
    change energy thresh upon dir
    """

    # change thresh in db
    self.energy_thresh_db -= ud * 0.5

    # update render message
    self.interactable_dict['text_energy'].message = 'e: {:.1f}dB'.format(self.energy_thresh_db)
    self.interactable_dict['text_energy'].render()



class CanvasDevice(Canvas):
  """
  device selection canvas
  """

  def __init__(self, screen, mic, size, position):

    # Parent init
    super().__init__(screen, size=size, position=position)

    # arguments
    self.mic = mic

    # set background color
    self.color_background = self.color_bag.canvas_option_background

    # enable view
    self.enabled = False

    # device list texts
    self.text_device_list = []

    # device id dict
    self.device_id_dict = {}

    # update
    self.interactable_dict.update({'text_info': Text(self.canvas_surf, message='devices: ', position=(0, 0), font_size='small', color=self.color_bag.text_menu)})

    # active device number and id
    self.active_device_num = 0
    self.active_device_id = 0

    # devices
    self.devices_to_text()


  def devices_to_text(self):
    """
    input devices to text messages
    """

    # get devices
    device_dicts = self.mic.extract_devices()

    # selector
    text_indicator = Text(self.canvas_surf, message='*', position=(5, 40), font_size='tiny', color=self.color_bag.text_menu)

    for i, (num_device, device_dict) in enumerate(device_dicts.items()):

      # device text
      text = Text(self.canvas_surf, message=device_dict['name'], position=(25, i*25 + 40), font_size='tiny', color=self.color_bag.text_menu)

      # indicator position
      if num_device == self.mic.device:
        self.active_device_num = num_device
        self.active_device_id = i
        text_indicator.position = (5, i * 25 + 40)

      # device id dict update
      self.device_id_dict.update({i: num_device})

      # append to list
      self.text_device_list.append((num_device, text))

    # update text devices
    [self.interactable_dict.update({'text_device{}'.format(n):t}) for n, t in self.text_device_list]

    # update text indicator
    self.interactable_dict.update({'text_indicator':text_indicator})


  def device_select(self, active):
    """
    select device
    """

    # select color
    text_color = self.color_bag.text_menu_active if active else self.color_bag.text_menu

    # update active device
    self.active_device_num = self.device_id_dict[self.active_device_id]

    # change indicator position
    self.interactable_dict['text_indicator'].position = (5, self.active_device_id * 25 + 40)

    # set color
    self.interactable_dict['text_indicator'].color = text_color
    self.interactable_dict['text_device{}'.format(self.active_device_num)].color = text_color

    # render text
    self.interactable_dict['text_indicator'].render()
    self.interactable_dict['text_device{}'.format(self.active_device_num)].render()



class CanvasPrompt(Canvas):
  """
  prompt canvas
  """

  def __init__(self, screen, size=None, position=(0, 0)):

    # Parent init
    super().__init__(screen, size=size, position=position)

    # set background color
    self.color_background = self.color_bag.canvas_win_backgound

    # deselect
    self.enabled = False

    # update
    self.interactable_dict.update({
      'title': Text(self.canvas_surf, message='Prompt', position=(250, 75), font_size='big', color=self.color_bag.text_win), 
      'sub': Text(self.canvas_surf, message='press Enter', position=(250, 125), font_size='small', color=self.color_bag.text_win)})


  def reset(self):
    """
    reset level
    """

    # reenable
    self.enabled = False

    # interactables reset
    for interactable in self.interactable_dict.values(): interactable.reset()



class CanvasWin(CanvasPrompt):
  """
  win canvas
  """

  def __init__(self, screen, size=None, position=(0, 0)):

    # parent init
    super().__init__(screen, size=size, position=position)

    # change title
    self.interactable_dict['title'].change_message('Level Complete', position=(200, 75))
    self.interactable_dict['sub'].change_message('press Enter', position=(200, 125))

    # spaceship
    self.interactable_dict.update({'spaceship': Spaceship(surf=self.canvas_surf, position=(250, 175), scale=(2, 2), thing_type='empty')})


  def add_spaceship_part(self, thing_type=None):
    """
    add part of spaceship
    """
    if thing_type is not None: self.interactable_dict.update({'spaceship_engine': Spaceship(surf=self.canvas_surf, position=(250, 175), scale=(2, 2), thing_type=thing_type)})



class CanvasLoose(CanvasPrompt):
  """
  loose canvas (changed from win canvas)
  """

  def __init__(self, screen, size=None, position=(0, 0)):

    # parent init
    super().__init__(screen, size=size, position=position)

    # change title
    self.interactable_dict['title'].change_message('Loose', position=(250, 75))



class CanvasCredits(CanvasPrompt):
  """
  credits canvas
  """

  def __init__(self, screen, size=None, position=(0, 0)):

    # parent init
    super().__init__(screen, size=size, position=position)

    # change title
    self.interactable_dict['title'].change_message('Credits', position=(230, 75))
    self.interactable_dict.update({'sub': Interactable()})

    # my text
    self.interactable_dict.update({'text_chris': Text(self.canvas_surf, message='Christian Walter', position=(230, 175), font_size='small', color=self.color_bag.text_win)})
    self.interactable_dict.update({'text_tu': Text(self.canvas_surf, message='TU Graz', position=(230, 200), font_size='small', color=self.color_bag.text_win)})

    # spaceship
    self.interactable_dict.update({'spaceship': Spaceship(surf=self.canvas_surf, position=(250, 265), scale=(2, 2), thing_type='whole')})



if __name__ == '__main__':
  """
  main
  """

  import yaml

  from game_logic import GameLogic
  from input_handler import InputKeyHandler

  # yaml config file
  cfg = yaml.safe_load(open("../config.yaml"))


  # init pygame
  pygame.init()

  # init display
  screen = pygame.display.set_mode(cfg['game']['screen_size'])

  # menu
  #canvas = CanvasMainMenu(screen)
  #canvas = CanvasLoose(screen)
  #canvas = CanvasWin(screen)
  canvas = CanvasCredits(screen)
  canvas.enabled = True

  # game logic
  game_logic = GameLogic()

  # key handler
  input_handler_key = InputKeyHandler(objs=[game_logic]) 

  # add clock
  clock = pygame.time.Clock()

  # game loop
  while game_logic.run_loop:
    for event in pygame.event.get():

      # input handling
      input_handler_key.event_update(event)

      # canvas event
      canvas.event_update(event)

    # text update
    canvas.update()
    canvas.draw()

    # update display
    pygame.display.flip()

    # reduce frame rate
    clock.tick(cfg['game']['fps'])


  # end pygame
  pygame.quit()