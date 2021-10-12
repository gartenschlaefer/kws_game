"""
character class
"""

import pygame

from interactable import Interactable


class Text(Interactable):
  """
  character class
  """

  def __init__(self, surf, message, position, font_size='small', color=(0, 0, 0), enabled=True):

    # arguments
    self.surf = surf
    self.message = message
    self.position = position
    self.font_size = font_size
    self.color = color
    self.enabled = enabled

    # init font
    pygame.font.init()

    # fonts
    self.font = pygame.font.SysFont('Courier', self.get_font_size(self.font_size))

    # render message
    self.render()


  def reset(self):
    """
    reset: enable it in case it was disabled once
    """
    self.enabled = True


  def set_active(self, active):
    """
    set active
    """
    self.enabled = active


  def get_font_size(self, font_size):
    """
    define font sizes
    """

    if font_size == 'big': return 40
    if font_size == 'normal': return 30
    elif font_size == 'small': return 20
    elif font_size == 'tiny_small': return 16
    elif font_size == 'tiny': return 11
    elif font_size == 'micro': return 9

    return 20


  def render(self):
    """
    render message
    """
    self.rendered_message = self.font.render(self.message, True, self.color)


  def change_message(self, message, position=None):
    """
    change message
    """
    self.message = message
    self.position = position if position is not None else self.position
    self.render()


  def draw(self):
    """
    draw text
    """
    if not self.enabled: return
    self.surf.blit(self.rendered_message, self.position)



if __name__ == '__main__':
  """
  text
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

  # text module
  text = Text(screen, message='yea', position=(0, 0), font_size='big', color=(0, 0, 0))
 
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
      game_logic.event_update(event)
      input_handler_key.event_update(event)

    # frame update
    game_logic.update()

    # fill screen
    screen.fill((255, 255, 255))

    # text update
    text.draw()

    # update display
    pygame.display.flip()

    # reduce frame rate
    clock.tick(cfg['game']['fps'])


  # end pygame
  pygame.quit()




