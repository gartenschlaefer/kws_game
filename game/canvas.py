"""
canvas, space for creativity
"""

import pygame

from interactable import Interactable
from text import Text
from button import StartButton, EndButton, HelpButton, OptionButton, DeviceButton, ThreshButton, CmdButton
from color_bag import ColorBag
from mic_bar import MicBar


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
    self.interactable_dict.update({'text': text, 'start_button': StartButton(self.canvas_surf, position=(100, 100), scale=(3, 3)), 'help_button': HelpButton(self.canvas_surf, position=(100, 200), scale=(3, 3)), 'option_button': OptionButton(self.canvas_surf, position=(100, 300), scale=(3, 3)), 'end_button': EndButton(self.canvas_surf, position=(100, 400), scale=(3, 3))})



class CanvasHelpMenu(Canvas):
  """
  main menu canvas
  """

  def __init__(self, screen):

    # Parent init
    super().__init__(screen)

    # add text
    text = Text(self.canvas_surf, message='help', position=(0, 0), font_size='small', color=self.color_bag.text_menu)

    # update canvas objects
    self.interactable_dict.update({'text': text, 'end_button': EndButton(self.canvas_surf, position=(100, 300), scale=(3, 3))})



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
    text = Text(self.canvas_surf, message='option', position=(0, 0), font_size='small', color=self.color_bag.text_menu)

    # mic bar
    mic_bar = MicBar(self.canvas_surf, self.mic, position=(550, 200), bar_size=(30, 150), scale_margin=(50, 40))

    # device canvas
    self.interactable_dict.update({'device_canvas': CanvasDevice(self.canvas_surf, self.mic, size=(350, 350), position=(200, 50))})

    # thresh canvas
    self.interactable_dict.update({'thresh_canvas': CanvasThresh(self.canvas_surf, self.mic, size=(350, 350), position=(200, 50))})
    
    # command canvas
    self.interactable_dict.update({'cmd_canvas': CanvasCommand(self.canvas_surf, self.mic, size=(350, 350), position=(200, 50))})

    # update canvas objects
    self.interactable_dict.update({'text': text, 'mic_bar': mic_bar, 'cmd_button': CmdButton(self.canvas_surf, position=(50, 75), scale=(3, 3)), 'thresh_button': ThreshButton(self.canvas_surf, position=(50, 175), scale=(3, 3)), 'device_button': DeviceButton(self.canvas_surf, position=(50, 275), scale=(3, 3)), 'end_button': EndButton(self.canvas_surf, position=(50, 375), scale=(3, 3))})



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

    # info text
    self.interactable_dict.update({'text_info': Text(self.canvas_surf, message='speech command: ', position=(0, 0), font_size='small', color=self.color_bag.text_menu)})



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

    # info text
    self.interactable_dict.update({'text_info': Text(self.canvas_surf, message='energy threshold: ', position=(0, 0), font_size='small', color=self.color_bag.text_menu)})



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



class CanvasWin(Canvas):
  """
  win canvas
  """

  def __init__(self, screen, size=None, position=(0, 0)):

    # Parent init
    super().__init__(screen, size=size, position=position)

    # set background color
    self.color_background = self.color_bag.canvas_win_backgound

    # deselect
    self.enabled = False

    # info text
    win_title = Text(self.canvas_surf, message='Win', position=(275, 75), font_size='big', color=self.color_bag.text_win)
    win_sub = Text(self.canvas_surf, message='press Enter', position=(250, 125), font_size='small', color=self.color_bag.text_win)

    # device list texts
    self.text_device_list = []

    # update
    self.interactable_dict.update({'win_title': win_title, 'win_sub': win_sub})


  def reset(self):
    """
    reset level
    """

    # reenable
    self.enabled = False

    # interactables reset
    for interactable in self.interactable_dict.values(): interactable.reset()



if __name__ == '__main__':
  """
  main
  """

  import yaml

  from game_logic import GameLogic

  # yaml config file
  cfg = yaml.safe_load(open("../config.yaml"))


  # init pygame
  pygame.init()

  # init display
  screen = pygame.display.set_mode(cfg['game']['screen_size'])

  # menu
  canvas = CanvasMainMenu(screen)


  # game logic
  game_logic = GameLogic()

  # add clock
  clock = pygame.time.Clock()

  # game loop
  while game_logic.run_loop:
    for event in pygame.event.get():

      # input handling
      game_logic.event_update(event)
      canvas.event_update(event)

    # frame update
    game_logic.update()

    # text update
    canvas.update()

    # update display
    pygame.display.flip()

    # reduce frame rate
    clock.tick(cfg['game']['fps'])


  # end pygame
  pygame.quit()