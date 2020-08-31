"""
things class
"""

import pygame
from text import Text


class Thing(pygame.sprite.Sprite):
  """
  character class
  """

  def __init__(self, position, scale=(2, 2)):

    # MRO check
    super().__init__()

    # load image and create rect
    self.image = pygame.image.load("./art/thing.png").convert_alpha()
    self.rect = self.image.get_rect()

    # proper scaling
    self.image = pygame.transform.scale(self.image, (self.rect.width * scale[0], self.rect.height * scale[1]))
    self.rect = self.image.get_rect()

    # set rect position
    self.rect.x = position[0]
    self.rect.y = position[1]

    # interactions
    self.is_active = True


if __name__ == '__main__':
  """
  test character
  """

  from color_bag import ColorBag
  from levels import LevelThings
  from text import Text

  from game_logic import ThingsGameLogic

  # size of display
  screen_size = width, height = 640, 480

  # grid
  pixel_size = (20, 20)

  # init pygame
  pygame.init()

  # init display
  screen = pygame.display.set_mode(screen_size)

  # collection of game colors
  color_bag = ColorBag()
  text = Text(screen, color_bag)


  # level creation
  level = LevelThings(screen, screen_size, color_bag)

  # game logic with dependencies
  game_logic = ThingsGameLogic(level.henry, text)


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
    game_logic.update()
    level.update()
    text.update()

    # update display
    pygame.display.flip()

    # reduce framerate
    clock.tick(60)


  # end pygame
  pygame.quit()




