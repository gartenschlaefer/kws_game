"""
things class
"""

import pygame
import pathlib

from text import Text


class Thing(pygame.sprite.Sprite):
  """
  character class
  """

  def __init__(self, position, scale=(2, 2)):

    # MRO check
    super().__init__()

    # vars
    self.position = position
    self.scale = scale

    # load image and create rect
    self.image = pygame.image.load(str(pathlib.Path(__file__).parent.absolute()) + "/art/thing.png").convert_alpha()
    self.rect = self.image.get_rect()

    # proper scaling
    self.image = pygame.transform.scale(self.image, (self.rect.width * scale[0], self.rect.height * scale[1]))
    self.rect = self.image.get_rect()

    # set rect position
    self.rect.x = position[0]
    self.rect.y = position[1]

    # interactions
    self.is_active = True


  def set_position(self, position, is_init_pos=False):
    """
    set position absolute
    """

    # set internal pos
    self.position = position

    # set rect
    self.rect.x = self.position[0]
    self.rect.y = self.position[1]


if __name__ == '__main__':
  """
  test character
  """
  import yaml

  from levels import LevelThings, Level_01
  from text import Text
  from game_logic import ThingsGameLogic


  # yaml config file
  cfg = yaml.safe_load(open("../config.yaml"))

  # grid
  pixel_size = (20, 20)

  # init pygame
  pygame.init()

  # init display
  screen = pygame.display.set_mode(cfg['game']['screen_size'])

  # level creation
  levels = [LevelThings(screen, cfg['game']['screen_size']), Level_01(screen, cfg['game']['screen_size'])]

  # choose level
  level = levels[0]

  # game logic with dependencies
  game_logic = ThingsGameLogic(level, levels)

  # add clock
  clock = pygame.time.Clock()


  # game loop
  while game_logic.run_loop:
    for event in pygame.event.get():
      if event.type == pygame.QUIT: 
        run_loop = False

      # input handling
      game_logic.event_update(event)
      level.event_update(event)

    # frame update
    level = game_logic.update()
    level.update()

    # update display
    pygame.display.flip()

    # reduce framerate
    clock.tick(cfg['game']['fps'])


  # end pygame
  pygame.quit()




