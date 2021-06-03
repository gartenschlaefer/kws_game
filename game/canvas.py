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
    text = Text(self.canvas_surf)
    text.render_small_msg('main menu', (0, 0))

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
    text = Text(self.canvas_surf)
    text.render_small_msg('help', (0, 0))

    # update canvas objects
    self.interactable_dict.update({'text': text, 'end_button': EndButton(self.canvas_surf, position=(100, 300), scale=(3, 3))})



class CanvasOptionMenu(Canvas):
  """
  main menu canvas
  """

  def __init__(self, screen, mic):

    # Parent init
    super().__init__(screen)

    # arguments
    self.mic = mic

    # add text
    text = Text(self.canvas_surf)
    text.render_small_msg('option', (0, 0))

    # mic bar
    mic_bar = MicBar(self.canvas_surf, self.mic, position=(575, 200), color_bag=self.color_bag, size=(30, 150))

    # device canvas
    self.interactable_dict.update({'device_canvas': CanvasDevice(self.canvas_surf, self.mic, size=(350, 400), position=(200, 50))})

    # update canvas objects
    self.interactable_dict.update({'text': text, 'mic_bar': mic_bar, 'cmd_button': CmdButton(self.canvas_surf, position=(50, 75), scale=(3, 3)), 'thresh_button': ThreshButton(self.canvas_surf, position=(50, 175), scale=(3, 3)), 'device_button': DeviceButton(self.canvas_surf, position=(50, 275), scale=(3, 3)), 'end_button': EndButton(self.canvas_surf, position=(50, 375), scale=(3, 3))})



class CanvasDevice(Canvas):

  def __init__(self, screen, mic, size, position):

    # Parent init
    super().__init__(screen, size=size, position=position)

    # arguments
    self.mic = mic

    # set background color
    self.color_background = self.color_bag.canvas_device_background

    # deselect
    self.enabled = False

    # info text
    text_device_info = Text(self.canvas_surf)
    text_device_info.render_small_msg('devices: ', (0, 0))

    # device list texts
    self.text_device_list = []

    # update
    self.interactable_dict.update({'text_device_info': text_device_info})

    # devices
    self.devices_to_text()


  def devices_to_text(self):
    """
    input devices to text messages
    """

    # get devices
    device_dicts = self.mic.extract_devices()

    # selector
    text_device_indicator = Text(self.canvas_surf)

    for i, (num_device, device_dict) in enumerate(device_dicts.items()):
      #print("device: ", device_dict.keys())
      text = Text(self.canvas_surf)
      text.render_tiny_msg(device_dict['name'], (25, i*25 + 40))

      # indicator
      if i == self.mic.device: text_device_indicator.render_tiny_msg('*', (5, i*25 + 40))

      # append to list
      self.text_device_list.append(text)

    # update text devices
    [self.interactable_dict.update({'text_device{}'.format(i):t}) for i, t in enumerate(self.text_device_list)]

    # update text indicator
    self.interactable_dict.update({'text_device_indicator':text_device_indicator})



  def update(self):
    """
    update
    """
    
    if not self.enabled: return

    # update all interactables
    for interactable in self.interactable_dict.values(): interactable.update()




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