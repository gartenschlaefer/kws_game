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
  from levels import setup_level_square
  from character import Character
  from grid_world import GridWorld
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

  # sprite groups
  all_sprites = pygame.sprite.Group()
  wall_sprites = pygame.sprite.Group()
  thing_sprites = pygame.sprite.Group()

  # create game objects
  henry = Character(position=(width//2, height//2), scale=(2, 2))
  thing = Thing(position=(10*pixel_size[0], 10*pixel_size[1]), scale=(2, 2))
  grid_world = GridWorld(screen_size, color_bag, pixel_size=pixel_size)

  # game logic with dependencies
  game_logic = ThingsGameLogic(henry, text)

  # setup stuff
  setup_level_square(grid_world)

  # add to sprite groups
  all_sprites.add(henry, thing, grid_world.wall_sprites)
  thing_sprites.add(thing)

  # henry sees walls and things
  henry.walls = grid_world.wall_sprites
  henry.things = thing_sprites

  # add clock
  clock = pygame.time.Clock()


  # game loop
  while game_logic.run_loop:
    for event in pygame.event.get():
      if event.type == pygame.QUIT: 
        run_loop = False

      # input handling of henry
      henry.input_handler.handle(event)
      game_logic.event_update(event)

    # game logic
    game_logic.update()
    

    # update without drawing
    all_sprites.update()

    # fill screen
    screen.fill(color_bag.background)

    # draw sprites
    all_sprites.draw(screen)
    text.update()

    # update display
    pygame.display.flip()

    # reduce framerate
    clock.tick(60)


  # end pygame
  pygame.quit()




