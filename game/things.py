"""
character class
"""

import pygame



class Thing(pygame.sprite.Sprite):
  """
  character class
  """

  def __init__(self, position, scale=(3, 3)):

    # MRO check
    super().__init__()

    # load image and create rect
    self.image = pygame.image.load("./art/henry_front.png").convert_alpha()
    self.rect = self.image.get_rect()

    # proper scaling
    self.image = pygame.transform.scale(self.image, (self.rect.width * scale[0], self.rect.height * scale[1]))
    self.rect = self.image.get_rect()

    # set rect position
    self.rect.x = position[0]
    self.rect.y = position[1]

    # interactions
    self.walls = None
    self.is_active = True






if __name__ == '__main__':
  """
  test character
  """

  from color_bag import ColorBag
  from levels import setup_level_square
  from character import Character
  from grid_world import GridWorld

  # size of display
  screen_size = width, height = 640, 480

  # some vars
  run_loop = True

  # collection of game colors
  color_bag = ColorBag()

  # init pygame
  pygame.init()

  # init display
  screen = pygame.display.set_mode(screen_size)

  # sprite groups
  all_sprites = pygame.sprite.Group()
  wall_sprites = pygame.sprite.Group()

  # create the character
  henry = Character(position=(width//2, height//2), scale=(2, 2))

  # create gridworld
  grid_world = GridWorld(screen_size, color_bag)
  setup_level_square(grid_world)

  # add to sprite groups
  all_sprites.add(henry, grid_world.wall_sprites)

  # henry sees walls
  henry.walls = grid_world.wall_sprites

  # add clock
  clock = pygame.time.Clock()

  # game loop
  while run_loop:
    for event in pygame.event.get():
      if event.type == pygame.QUIT: 
        run_loop = False

      # input handling of henry
      henry.input_handler.handle(event)

    # update sprites
    all_sprites.update()

    # fill screen
    screen.fill(color_bag.background)

    # draw sprites
    all_sprites.draw(screen)

    # update display
    pygame.display.flip()

    # reduce framerate
    clock.tick(60)


  # end pygame
  pygame.quit()




