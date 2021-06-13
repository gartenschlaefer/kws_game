"""
character class
"""

import pygame

from interactable import Interactable


class Text(Interactable):
  """
  character class
  """

  def __init__(self, surf, message, position, font_size='small', color=(0, 0, 0)):

    # arguments
    self.surf = surf
    self.message = message
    self.position = position
    self.font_size = font_size
    self.color = color

    # init font
    pygame.font.init()

    # fonts
    self.font = pygame.font.SysFont('Courier', self.get_font_size(self.font_size))

    # render message
    self.render()


  def get_font_size(self, font_size):
    """
    define font sizes
    """

    if font_size == 'big': return 40
    elif font_size == 'small': return 20
    elif font_size == 'tiny': return 11
    elif font_size == 'micro': return 9

    return 20


  def render(self):
    """
    render message
    """
    self.rendered_message = self.font.render(self.message, True, self.color)


  def draw(self):
    """
    draw text
    """
    self.surf.blit(self.rendered_message, self.position)



if __name__ == '__main__':
  """
  text
  """

  import yaml

  from game_logic import GameLogic
  
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

  # add clock
  clock = pygame.time.Clock()

  # game loop
  while game_logic.run_loop:
    for event in pygame.event.get():

      # input handling
      game_logic.event_update(event)

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




