"""
character class
"""

import pygame

from color_bag import ColorBag
from interactable import Interactable


class Text(Interactable):
  """
  character class
  """

  def __init__(self, surf):

    # arguments
    self.surf = surf

    # colorbag
    self.color_bag = ColorBag()

    # init font
    pygame.font.init()

    # fonts
    self.big_font = pygame.font.SysFont('Courier', 40)
    self.small_font = pygame.font.SysFont('Courier', 20)

    # messages
    self.big_msg = None
    self.small_msg = None

    # positions
    self.big_pos = None
    self.small_pos = None


  def render_small_msg(self, msg, pos):
    """
    render a small message
    """

    self.small_msg = self.small_font.render(msg, True, self.color_bag.win)
    self.small_pos = pos


  def win_message(self, big_pos=(200, 100), small_pos=(200, 150)):
    """
    write win message
    """

    self.big_msg = self.big_font.render('Win', True, self.color_bag.win)
    self.small_msg = self.small_font.render('press Enter', True, self.color_bag.win)
    self.big_pos = big_pos
    self.small_pos = small_pos


  def reset(self):
    """
    reset message
    """

    self.big_msg = None
    self.small_msg = None


  def update(self):
    """
    update texts on surface
    """
    pass


  def draw(self):
    """
    draw text
    """

    # draw messages
    if self.big_msg is not None: self.surf.blit(self.big_msg, self.big_pos)
    if self.small_msg is not None: self.surf.blit(self.small_msg, self.small_pos)



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
  text = Text(screen)
  text.win_message()
 

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
    screen.fill(text.color_bag.background)

    # text update
    text.draw()

    # update display
    pygame.display.flip()

    # reduce frame rate
    clock.tick(cfg['game']['fps'])


  # end pygame
  pygame.quit()




