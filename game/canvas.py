"""
canvas, space for creativity
"""

import pygame

from interactable import Interactable
from text import Text
from button import StartButton, EndButton, HelpButton, OptionButton
from color_bag import ColorBag


class Canvas(Interactable):
  """
  Canvas base class
  """

  def __init__(self, screen):

    # arguments
    self.screen = screen

    # canvas surface (use same size as screen)
    self.canvas_surf = pygame.Surface(self.screen.get_size())

    # colorbag
    self.color_bag = ColorBag()

    # interactables
    self.interactables_dict = {}


  def reset(self):
    """
    reset level
    """

    # interactables
    for interactable in self.interactables_dict.values(): interactable.reset()


  def event_update(self, event):
    """
    event update
    """

    # interactables
    for interactable in self.interactables_dict.values(): interactable.event_update(event)


  def update(self):
    """
    update
    """
    
    # update all interactables
    for interactable in self.interactables_dict.values(): interactable.update()

    # fill screen
    self.canvas_surf.fill(self.color_bag.canvas_background)

    # blit stuff
    for interactable in self.interactables_dict.values(): interactable.draw()

    # draw canvas
    self.screen.blit(self.canvas_surf, (0, 0))



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
    self.interactables_dict.update({'text': text, 'start_button': StartButton(self.canvas_surf, position=(100, 100), scale=(3, 3)), 'help_button': HelpButton(self.canvas_surf, position=(100, 200), scale=(3, 3)), 'option_button': OptionButton(self.canvas_surf, position=(100, 300), scale=(3, 3)), 'end_button': EndButton(self.canvas_surf, position=(100, 400), scale=(3, 3))})



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
    self.interactables_dict.update({'text': text, 'end_button': EndButton(self.canvas_surf, position=(100, 300), scale=(3, 3))})



class CanvasOptionMenu(Canvas):
  """
  main menu canvas
  """

  def __init__(self, screen):

    # Parent init
    super().__init__(screen)

    # add text
    text = Text(self.canvas_surf)
    text.render_small_msg('option', (0, 0))

    # update canvas objects
    self.interactables_dict.update({'text': text, 'end_button': EndButton(self.canvas_surf, position=(100, 300), scale=(3, 3))})


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